"""Ensemble testleri - ozellikle 'OLS = cosine-optimal' iddiasi."""
import numpy as np
import pytest

from src.evaluation.metrics import cosine_similarity
from src.models.ensemble import CosineOptimalEnsemble, evaluate_ensemble_gain

rng = np.random.default_rng(7)


def test_ols_weights_are_cosine_optimal():
    """Rastgele agirliklar OLS cozumunu gecememeli (negatif serbest halde)."""
    y = rng.normal(0, 1, 500)
    P = np.column_stack([y + rng.normal(0, 1, 500) for _ in range(3)])
    ens = CosineOptimalEnsemble(["a", "b", "c"], non_negative=False).fit(P, y)
    best = cosine_similarity(y, ens.predict(P))
    for _ in range(200):
        w = rng.normal(0, 1, 3)
        assert cosine_similarity(y, P @ w) <= best + 1e-9


def test_weights_sum_to_one_in_absolute_value():
    y = rng.normal(0, 1, 300)
    P = np.column_stack([y + rng.normal(0, 0.5, 300) for _ in range(2)])
    ens = CosineOptimalEnsemble(["a", "b"]).fit(P, y)
    assert np.abs(ens.weights_).sum() == pytest.approx(1.0)


def test_non_negative_weights_are_non_negative():
    y = rng.normal(0, 1, 300)
    P = np.column_stack([y + rng.normal(0, 0.5, 300), -y + rng.normal(0, 0.5, 300)])
    ens = CosineOptimalEnsemble(["good", "bad"], non_negative=True).fit(P, y)
    assert (ens.weights_ >= -1e-12).all()


def test_ensemble_of_identical_models_gains_nothing():
    y = rng.normal(0, 1, 400)
    p = y + rng.normal(0, 0.5, 400)
    out = evaluate_ensemble_gain({"m1": p, "m2": p.copy()}, y)
    assert out["gain"] == pytest.approx(0.0, abs=1e-9)
    assert not out["beats_best_single"]


def test_ensemble_helps_with_complementary_errors():
    n = 800
    s1, s2 = rng.normal(0, 1, n), rng.normal(0, 1, n)
    y = s1 + s2
    out = evaluate_ensemble_gain({"m1": s1 + rng.normal(0, .1, n),
                                  "m2": s2 + rng.normal(0, .1, n)}, y)
    assert out["beats_best_single"] and out["gain"] > 0.1


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        CosineOptimalEnsemble(["a"]).fit(np.zeros((10, 1)), np.zeros(11))

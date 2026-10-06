"""The extra ensemble members and the probe's blend logic.

CatBoost and PyTorch are not CI dependencies, so their tests skip when the package is missing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from src.models import ensemble_probe as ep
from src.models import extra_members as em


def _data(n: int = 1500, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 5)).astype("float32"), columns=list("abcde"))
    X.loc[::13, "b"] = np.nan
    X.loc[::29, "c"] = np.inf
    signal = 0.5 * X["a"].to_numpy() - 0.3 * np.nan_to_num(X["d"].to_numpy())
    y = 0.002 * signal + rng.standard_t(3, n) * 0.002  # heavy tailed, small scale like the real target
    return X, y


# ------------------------------------------------------------------ LightGBM variants
def test_huber_threshold_is_set_from_the_target_scale_not_a_constant():
    X, y = _data()
    m = em.HuberLightGBM(num_boost_round=20, early_stopping_rounds=5)
    m.fit(X, y, eval_set=(X, y))
    assert m.params["objective"] == "huber"
    assert m.params["alpha"] == pytest.approx(em.HUBER_DELTAS * float(np.std(y)))
    assert m.predict(X).shape == (len(X),)


def test_extra_trees_variant_differs_from_the_default_learner():
    m = em.ExtraTreesLightGBM(num_boost_round=20, early_stopping_rounds=5)
    assert m.params["extra_trees"] is True and m.params["num_leaves"] == 31
    X, y = _data()
    assert np.isfinite(m.fit(X, y, eval_set=(X, y)).predict(X)).all()


def test_the_variants_report_their_own_names():
    assert em.HuberLightGBM().name == "lightgbm_huber"
    assert em.ExtraTreesLightGBM().name == "lightgbm_extra"


# ------------------------------------------------------------------ CatBoost and MLP
def test_catboost_fits_predicts_and_handles_nan_and_inf():
    pytest.importorskip("catboost")
    X, y = _data()
    m = em.CatBoostModel(iterations=30, depth=4, early_stopping_rounds=5)
    pred = m.fit(X, y, eval_set=(X, y)).predict(X)
    assert pred.shape == (len(X),) and np.isfinite(pred).all()


def test_mlp_predicts_on_the_targets_own_scale_and_keeps_column_order():
    pytest.importorskip("torch")
    X, y = _data()
    m = em.MLPModel(epochs=3, patience=3, batch=256, hidden=(16, 8), chunk_rows=400)
    pred = m.fit(X, y, eval_set=(X, y)).predict(X)
    assert pred.shape == (len(X),) and np.isfinite(pred).all()
    # The unit is the target's own: the network works in standard deviations internally, and
    # predict() multiplies that back. A target this noisy is predicted with a small amplitude,
    # so the check is on the conversion itself, not on how big the output happens to be.
    assert m.scale_ == pytest.approx(float(np.std(y)))
    np.testing.assert_allclose(pred, m._forward(m._array(X)) * m.scale_, rtol=1e-9)
    assert 0.0 < float(np.std(pred)) < float(np.std(y))
    np.testing.assert_allclose(m.predict(X[list(reversed(X.columns))]), pred, atol=1e-6)


def test_mlp_prediction_does_not_depend_on_the_chunk_size():
    pytest.importorskip("torch")
    X, y = _data()
    m = em.MLPModel(epochs=2, patience=2, batch=256, hidden=(16, 8), chunk_rows=300)
    m.fit(X, y, eval_set=(X, y))
    a = m.predict(X)
    m.chunk_rows = 77
    np.testing.assert_allclose(m.predict(X), a, atol=1e-6)


def test_unfitted_members_refuse_to_predict():
    with pytest.raises(RuntimeError, match="fit"):
        em.CatBoostModel().predict(pd.DataFrame({"a": [1.0]}))
    with pytest.raises(RuntimeError, match="fit"):
        em.MLPModel().predict(pd.DataFrame({"a": [1.0]}))


# ------------------------------------------------------------------ the probe's blend logic
def test_a_member_with_independent_signal_improves_the_nnls_blend():
    rng = np.random.default_rng(0)
    n = 6000
    s1, s2 = rng.normal(size=n), rng.normal(size=n)
    y = s1 + s2 + rng.normal(scale=2.0, size=n)
    val = {"a": s1[:3000] + 0.0, "b": s2[:3000] + 0.0}
    test = {"a": s1[3000:] + 0.0, "b": s2[3000:] + 0.0}
    from src.evaluation.metrics import cosine_similarity

    only_a, _ = ep.blend(("a",), val, test, y[:3000])
    both, w = ep.blend(("a", "b"), val, test, y[:3000])
    assert cosine_similarity(y[3000:], both) > cosine_similarity(y[3000:], only_a)
    assert w["a"] > 0 and w["b"] > 0


def test_a_member_that_is_pure_noise_gets_little_weight():
    rng = np.random.default_rng(1)
    n = 6000
    s = rng.normal(size=n)
    y = s + rng.normal(scale=2.0, size=n)
    val = {"a": s[:3000], "noise": rng.normal(size=3000)}
    test = {"a": s[3000:], "noise": rng.normal(size=3000)}
    _, w = ep.blend(("a", "noise"), val, test, y[:3000])
    assert w["noise"] < 0.1 * w["a"]


def test_the_verdict_needs_both_a_clear_interval_and_the_shipping_size():
    noise = np.random.default_rng(0).normal(0, 0.0003, 23)
    clear_but_tiny = ep.verdict(noise + 0.0004)
    assert clear_but_tiny["adds_signal"] and not clear_but_tiny["worth_shipping"]
    big = ep.verdict(noise + 0.003)
    assert big["adds_signal"] and big["worth_shipping"]
    flat = ep.verdict(np.random.default_rng(1).normal(0, 0.004, 23))
    assert not flat["adds_signal"] and "not shown" in flat["verdict"]
    assert flat["ship_gain_threshold"] == 0.0010


def test_the_members_named_in_the_probe_all_have_a_factory():
    for name in (*ep.BASE, *ep.EXTRA):
        if name == "catboost":
            pytest.importorskip("catboost")
        if name == "mlp":
            pytest.importorskip("torch")
        assert ep.make_member(name).name == name


# ------------------------------------------------------------------ MLP preprocessing, no torch
def _fitted_preprocessing(chunk_rows: int):
    X, _ = _data()
    m = em.MLPModel(chunk_rows=chunk_rows)
    m.features_ = em.feature_columns(X)
    m.imputer.fit(X, columns=m.features_)
    m._fit_scaler(X)
    return X, m


def test_mlp_standardises_with_the_training_mean_and_std_and_clips():
    X, m = _fitted_preprocessing(chunk_rows=400)
    A = m._array(X)
    assert A.dtype == np.float32 and np.isfinite(A).all()  # NaN and inf were imputed away
    assert np.abs(A).max() <= 6.0 + 1e-6
    assert np.abs(A.mean(axis=0)).max() < 0.1  # imputed medians sit near the mean on symmetric data
    assert np.allclose(A.std(axis=0), 1.0, atol=0.15)


def test_mlp_preprocessing_does_not_depend_on_the_chunk_size():
    X, small = _fitted_preprocessing(chunk_rows=97)
    _, big = _fitted_preprocessing(chunk_rows=5000)
    np.testing.assert_allclose(small.mu_, big.mu_, rtol=1e-9)
    np.testing.assert_allclose(small.sd_, big.sd_, rtol=1e-9)
    np.testing.assert_array_equal(small._array(X), big._array(X))


def test_mlp_scales_a_constant_column_without_dividing_by_zero():
    X, _ = _data()
    X["const"] = np.float32(7.0)
    m = em.MLPModel(chunk_rows=500)
    m.features_ = em.feature_columns(X)
    m.imputer.fit(X, columns=m.features_)
    m._fit_scaler(X)
    A = m._array(X)
    assert np.isfinite(A).all() and np.all(A[:, m.features_.index("const")] == 0.0)


# ------------------------------------------------------------------ CatBoost wiring, fake package
class _FakePool:
    def __init__(self, data, label=None):
        self.data, self.label = data, label


class _FakeRegressor:
    last: _FakeRegressor | None = None

    def __init__(self, **kw):
        self.kw, self.fit_kw, self.fit_columns = kw, {}, None
        _FakeRegressor.last = self

    def fit(self, pool, **kw):
        self.fit_columns, self.fit_kw = list(pool.data.columns), kw
        return self

    def predict(self, X):
        return np.zeros(len(X))


@pytest.fixture
def fake_catboost(monkeypatch):
    import sys
    import types

    mod = types.ModuleType("catboost")
    mod.CatBoostRegressor, mod.Pool = _FakeRegressor, _FakePool  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "catboost", mod)


def test_catboost_wrapper_trains_on_features_only_and_early_stops_on_the_eval_set(fake_catboost):
    X, y = _data()
    X["month"], X["target"] = 0, 0.0  # columns a model must never see
    m = em.CatBoostModel(iterations=10, early_stopping_rounds=7, seed=3)
    m.fit(X, y, eval_set=(X, y))
    fake = _FakeRegressor.last
    assert fake is not None
    assert fake.fit_columns == list("abcde")  # month and target stay out
    assert fake.kw["random_seed"] == 3 and fake.kw["allow_writing_files"] is False
    assert fake.fit_kw["use_best_model"] is True and fake.fit_kw["early_stopping_rounds"] == 7
    assert m.predict(X).shape == (len(X),)


def test_catboost_wrapper_without_an_eval_set_does_not_ask_for_early_stopping(fake_catboost):
    X, y = _data()
    em.CatBoostModel().fit(X, y)
    fake = _FakeRegressor.last
    assert fake is not None and fake.fit_kw == {}

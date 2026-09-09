"""The artefact the serving layer loads must behave like the model that was fitted.

This path has produced two real defects, both silent:

  1. `ship.py` handed `save_bundle` the bare LightGBM booster while naming the artefact
     "ensemble", so /model-info described a blend and /predict returned single-model
     predictions.
  2. Unwrapping RidgeModel to its inner sklearn estimator dropped the median imputation,
     and predict() raised `Input X contains NaN` on the first gap in the test set - a
     failure invisible on the training fold, which has no gaps.

Neither had a test. Both would fail these.

Everything runs on a small synthetic frame: the real path needs a 1.3 GB feature table
and hours of training, which is exactly the reason it went untested.
"""
from __future__ import annotations

import builtins
import sys

import numpy as np
import pandas as pd
import pytest
from src.inference.ensemble import load_ensemble, save_ensemble
from src.inference.predictor import Predictor, load_bundle, save_bundle
from src.models.baseline import RidgeModel
from src.models.lightgbm_model import LightGBMModel
from src.models.xgboost_model import XGBoostModel

N_ROWS, N_FEATURES = 400, 8
NON_FEATURES = ("sample_id", "month", "target")

# Tolerance for "the artefact reproduces the fitted model".
#
# Both boosters accumulate in float32, and a round trip through a model file can change
# the order those terms are summed - so a handful of rows differ in the last bits. CI on
# Linux showed 2 rows in 400 differing by 1.5e-09 absolute, 6.6e-06 relative, against
# predictions of order 0.06. That is the arithmetic, not the model.
#
# 1e-4 is still two orders of magnitude tighter than any defect this is guarding against:
# serving one member instead of the blend moves predictions by tens of percent, and
# dropping the ridge term (weight 0.1) by more.
RTOL, ATOL = 1e-4, 1e-9


@pytest.fixture(scope="module")
def data():
    """A frame shaped like the real one: sample_id, month, target, then features."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(N_ROWS, N_FEATURES))
    y = 0.4 * X[:, 0] - 0.3 * X[:, 1] + rng.normal(scale=0.5, size=N_ROWS)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(N_FEATURES)])
    df.insert(0, "sample_id", np.arange(N_ROWS))
    df.insert(1, "month", np.repeat(np.arange(8), N_ROWS // 8))
    df["target"] = y
    return df


@pytest.fixture(scope="module")
def features(data):
    return [c for c in data.columns if c not in NON_FEATURES]


@pytest.fixture(scope="module")
def fitted(data):
    """The three base models ship.py builds, fitted on the synthetic frame."""
    y = data["target"].to_numpy()
    models = {
        "lightgbm": LightGBMModel(num_boost_round=20, early_stopping_rounds=None),
        "xgboost": XGBoostModel(num_boost_round=20, early_stopping_rounds=None),
        "ridge": RidgeModel(alpha=1.0),
    }
    for m in models.values():
        m.fit(data, y)
    return models


@pytest.fixture(scope="module")
def weights():
    return {"lightgbm": 0.6, "xgboost": 0.3, "ridge": 0.1}


@pytest.fixture()
def artefact_dir(tmp_path_factory, fitted, weights, features):
    d = tmp_path_factory.mktemp("shipped")
    save_ensemble(d, models=fitted, weights=weights, features=features)
    return d


def blend_of(fitted, weights, frame):
    """What the fitted wrappers predict, combined by hand - the reference answer."""
    return sum(w * fitted[n].predict(frame) for n, w in weights.items())


# ------------------------------------------------------------------ the round trip

def test_the_loaded_artefact_reproduces_the_fitted_blend(artefact_dir, fitted, weights,
                                                         data):
    """Saved and reloaded, it must predict what the fitted wrappers predict.

    This is the property the whole serving layer rests on, and nothing checked it.
    """
    got = load_ensemble(artefact_dir).predict(data)
    np.testing.assert_allclose(got, blend_of(fitted, weights, data), rtol=RTOL, atol=ATOL)


def test_it_loads_without_importing_the_training_package(artefact_dir, monkeypatch):
    """The API image ships without src.models; loading must not need it.

    Enforced by breaking the import rather than by trusting the code to be careful:
    pickling a fitted wrapper would drag `src.models` in, and would pass a test that only
    inspected the file list.
    """
    real_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.startswith("src.models"):
            raise AssertionError(f"serving must not import {name}")
        return real_import(name, *args, **kwargs)

    for mod in [m for m in list(sys.modules) if m.startswith("src.models")]:
        monkeypatch.delitem(sys.modules, mod, raising=False)
    monkeypatch.setattr(builtins, "__import__", guarded)

    assert load_ensemble(artefact_dir) is not None


# ------------------------------------------------------------------ the NaN defect

def test_ridge_still_imputes_after_the_round_trip(artefact_dir, data):
    """The exact failure that reached the test set: a gap, and Ridge blows up.

    The medians travel with the artefact. Unwrapping RidgeModel to its sklearn estimator
    left them behind and raised `Input X contains NaN` on the first missing value.

    Gaps are the normal case, not an edge case: 5.9% of the real feature matrix is NaN,
    because the 1-second windows are empty for most samples.
    """
    gappy = data.copy()
    gappy.loc[gappy.index[:20], "f0"] = np.nan
    gappy.loc[gappy.index[5:10], "f3"] = np.nan

    out = load_ensemble(artefact_dir).predict(gappy)
    assert np.isfinite(out).all(), "a gap in the input produced a non-finite prediction"


def test_an_infinite_feature_fails_loudly_rather_than_being_absorbed(artefact_dir, data):
    """Infinity is a contract violation, and the serving layer should not paper over it.

    The feature contract already rejects non-finite values (src/data/validation.py), and
    the exported feature sample contains none - ratio features like Kyle's lambda could
    produce them on a division by zero, which is precisely the case that should stop the
    pipeline rather than reach a model. XGBoost refuses such input outright; this pins
    that as the intended behaviour so nobody "fixes" it into a silent imputation later.
    """
    broken = data.copy()
    broken.loc[broken.index[0], "f0"] = np.inf
    with pytest.raises(Exception, match="(?i)inf"):
        load_ensemble(artefact_dir).predict(broken)


def test_the_medians_are_the_ones_that_were_fitted(artefact_dir, fitted):
    """Imputing with the WRONG medians is silent - the numbers just drift."""
    stored = load_ensemble(artefact_dir).parts["ridge"]["medians"]
    assert stored == pytest.approx(fitted["ridge"].imputer.medians_.to_dict())


# ------------------------------------------------------------------ column order

def test_column_order_does_not_change_the_prediction(artefact_dir, data):
    """Every base model is positional. A reordered frame must not silently re-pair
    coefficients with the wrong features."""
    art = load_ensemble(artefact_dir)
    shuffled = data[list(reversed(data.columns))]
    np.testing.assert_allclose(art.predict(shuffled), art.predict(data), rtol=1e-9)


def test_a_missing_feature_raises_rather_than_being_filled_in(artefact_dir, data):
    art = load_ensemble(artefact_dir)
    with pytest.raises(KeyError):
        art.predict(data.drop(columns=["f2"]))


# ------------------------------------------------------------------ weights

def test_the_weights_are_actually_applied(tmp_path, fitted, features, data):
    """A blend that ignored its weights would still pass a round-trip test built on equal
    weights, so this pins a lopsided set against the single model it favours."""
    save_ensemble(tmp_path, models=fitted, features=features,
                  weights={"lightgbm": 1.0, "xgboost": 0.0, "ridge": 0.0})
    np.testing.assert_allclose(
        load_ensemble(tmp_path).predict(data), fitted["lightgbm"].predict(data), rtol=RTOL,
    )


# ------------------------------------------------------------------ through the bundle

def test_the_bundle_says_ensemble_and_serves_an_ensemble(tmp_path, fitted, weights, data,
                                                         features):
    """The defect this file was written for: the metadata and the model must agree.

    The artefact was named "ensemble" while holding one LightGBM booster, so /model-info
    and /predict described different models and nothing failed.
    """
    save_bundle(tmp_path, model={"models": fitted, "weights": weights}, kind="ensemble",
                features=features, name="ensemble", version="test")

    bundle = load_bundle(tmp_path)
    assert bundle.name == "ensemble"

    served = bundle.model.predict(data[features])
    np.testing.assert_allclose(served, blend_of(fitted, weights, data),
                               rtol=RTOL, atol=ATOL)

    # ...and it is NOT any single member, which is what the broken version served.
    assert not np.allclose(served, fitted["lightgbm"].predict(data), rtol=1e-3)


def test_predictor_serves_the_ensemble_from_dict_rows(tmp_path, fitted, weights, data,
                                                      features):
    """End to end on the API's own path: dict rows in, blended predictions out."""
    save_bundle(tmp_path, model={"models": fitted, "weights": weights}, kind="ensemble",
                features=features, name="ensemble", version="test")

    predictor = Predictor.from_dir(tmp_path)
    out = predictor.predict(data[features].head(5).to_dict("records"))

    # Looser still: the request path casts to float32 on the way in, so this tolerance is
    # the cast rather than the round trip.
    np.testing.assert_allclose(out, blend_of(fitted, weights, data.head(5)),
                               rtol=1e-3, atol=1e-7)
    assert predictor.info()["model_name"] == "ensemble"

"""Tests for the hyperparameter search.

The claim this module makes is not "these are good parameters" but "this much of the gain
is real". That claim rests on two mechanics: the seed genuinely resampling the noise (if it
did not, re-scoring would be a tautology and every gain would look real), and the search
never being allowed to see the folds it will later be judged on.
"""
import numpy as np
import pandas as pd
import pytest

from src.models.tuning import SEARCH_FOLDS, cv_score, suggest


def _frame(n_months=71, per_month=120, n_features=8, seed=0):
    """A small dataset with the real column contract and a faint learnable signal."""
    rng = np.random.default_rng(seed)
    n = n_months * per_month
    X = rng.normal(0, 1, (n, n_features)).astype("float32")
    target = (0.2 * X[:, 0] + 0.1 * X[:, 1] + rng.normal(0, 1, n)).astype("float32")
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(n_features)])
    df["month"] = np.repeat(np.arange(n_months), per_month).astype("int16")
    df["sample_id"] = np.arange(n, dtype="int32")
    df["target"] = target
    return df


class _FixedTrial:
    """Minimal stand-in for an optuna trial: always returns the low end of each range."""

    def suggest_float(self, name, lo, hi, log=False):
        return lo

    def suggest_int(self, name, lo, hi, log=False):
        return lo

    def suggest_categorical(self, name, choices):
        return choices[0]


def test_search_space_covers_every_tuned_parameter():
    params = suggest(_FixedTrial())
    assert set(params) == {
        "learning_rate", "num_leaves", "min_data_in_leaf", "feature_fraction",
        "bagging_fraction", "lambda_l1", "lambda_l2", "max_bin",
    }


def test_search_space_bounds_are_sane():
    """Bounds are a compute budget as much as a modelling choice.

    An earlier version reached num_leaves 511 / max_bin 255 / learning_rate 0.01, whose
    trials cost ~10x the baseline - the low learning rate being the trap, since early
    stopping never fires and every round runs. A one-hour search took three.
    """
    p = suggest(_FixedTrial())
    assert 0.02 <= p["learning_rate"] < 1, "low rates defeat early stopping"
    assert p["num_leaves"] >= 2
    assert 0 < p["feature_fraction"] <= 1
    assert 0 < p["bagging_fraction"] <= 1
    assert p["max_bin"] in (63, 127)


def test_search_space_never_suggests_a_leaking_parameter():
    """month and target are not features; nothing in the space may reintroduce them."""
    assert not {"month", "target", "sample_id"} & set(suggest(_FixedTrial()))


def test_cv_score_is_deterministic_for_a_fixed_seed():
    df = _frame()
    a, _ = cv_score(df, {}, folds=1, rounds=30, seed=7)
    b, _ = cv_score(df, {}, folds=1, rounds=30, seed=7)
    assert a == pytest.approx(b)


def test_the_seed_actually_resamples_the_noise():
    """The whole selection-optimism check depends on this.

    If the seed did not reach bagging and feature sampling, re-scoring the winner would
    reproduce the search score exactly and every gain would appear to survive.
    """
    df = _frame()
    a, _ = cv_score(df, {"bagging_fraction": 0.6, "feature_fraction": 0.6},
                    folds=1, rounds=30, seed=1)
    b, _ = cv_score(df, {"bagging_fraction": 0.6, "feature_fraction": 0.6},
                    folds=1, rounds=30, seed=2)
    assert a != b


def test_cv_score_returns_one_score_per_fold():
    df = _frame()
    mean, scores = cv_score(df, {}, folds=2, rounds=20, seed=0)
    assert len(scores) == 2
    assert mean == pytest.approx(float(np.mean(scores)))


def test_cv_score_uses_the_last_folds():
    """Folds are expanding-window, so the early ones train on far less data.

    Taking the last N is a deliberate choice; taking the first N would search against a
    protocol unlike the one used to report results.
    """
    df = _frame()
    _, three = cv_score(df, {}, folds=3, rounds=20, seed=0)
    _, one = cv_score(df, {}, folds=1, rounds=20, seed=0)
    assert one[0] == pytest.approx(three[-1])


def test_search_uses_fewer_folds_than_the_reporting_protocol():
    """The search runs on a cheaper protocol; the winner is confirmed on the full one."""
    from src.evaluation.temporal_validation import build_folds

    assert SEARCH_FOLDS < len(build_folds())


def test_defaults_are_reachable_as_an_empty_override():
    """An empty params dict must mean 'current defaults', or the baseline arm is wrong."""
    from src.models.lightgbm_model import DEFAULT_PARAMS

    df = _frame()
    score, _ = cv_score(df, {}, folds=1, rounds=20, seed=0)
    explicit, _ = cv_score(df, dict(DEFAULT_PARAMS), folds=1, rounds=20, seed=0)
    assert score == pytest.approx(explicit)

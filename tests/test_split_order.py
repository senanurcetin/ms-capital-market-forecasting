"""The sample-order check: it must tell a time series from the same values in random order."""
from __future__ import annotations

import numpy as np
import pandas as pd
from src.evaluation import split_order as so


def _slow_series(n: int = 20_000, seed: int = 0) -> pd.Series:
    """A positive series with slowly drifting level plus noise: regimes, like train."""
    rng = np.random.default_rng(seed)
    level = np.cumsum(rng.normal(0, 0.02, n))
    return pd.Series(np.exp(level + rng.normal(0, 0.5, n)))


def test_a_series_with_regimes_has_high_block_autocorrelation():
    ac, n = so.block_autocorr(_slow_series(), 500)
    assert n == 40 and ac > 0.6


def test_the_same_values_in_random_order_have_none():
    s = _slow_series()
    shuffled = pd.Series(np.random.default_rng(1).permutation(s.to_numpy()))
    ac, n = so.block_autocorr(shuffled, 500)
    assert abs(ac) < 3 / np.sqrt(n)


def test_non_positive_values_are_missing_not_zero():
    s = _slow_series().copy()
    s.iloc[::7] = 0.0
    ac, _ = so.block_autocorr(s, 500)
    assert np.isfinite(ac) and ac > 0.5


def test_too_few_blocks_returns_nan_rather_than_a_number():
    ac, n = so.block_autocorr(_slow_series(100), 50)
    assert n == 2 and np.isnan(ac)


def test_the_verdict_separates_an_ordered_train_from_an_unordered_test():
    train = pd.DataFrame({c: _slow_series(seed=i) for i, c in enumerate(so.SIGNALS)})
    unordered = pd.DataFrame({c: pd.Series(np.random.default_rng(9).permutation(
        _slow_series(seed=i + 5).to_numpy())) for i, c in enumerate(so.SIGNALS)})
    out = so.summarise(so.measure({"train": train, "test": unordered}))
    assert out["train_is_in_time_order"] and not out["test_is_in_time_order"]
    assert "cannot be computed on the test set" in out["verdict"]


def test_two_ordered_splits_are_reported_as_ordered():
    a = pd.DataFrame({c: _slow_series(seed=i) for i, c in enumerate(so.SIGNALS)})
    b = pd.DataFrame({c: _slow_series(seed=i + 3) for i, c in enumerate(so.SIGNALS)})
    out = so.summarise(so.measure({"train": a, "test": b}))
    assert out["verdict"] == "both splits are in time order"



def test_removing_the_group_mean_separates_order_between_groups_from_order_within_them():
    rng = np.random.default_rng(0)
    months = np.repeat(np.arange(30), 1000)
    level = np.repeat(rng.normal(0, 1.0, 30), 1000)  # months differ; inside one they do not
    s = pd.Series(np.exp(level + rng.normal(0, 0.3, len(months))))
    plain, _ = so.block_autocorr(s, 50)
    within, n = so.within_group_block_autocorr(s, pd.Series(months), 50)
    bias = -1 / (1000 / 50 - 1)  # removing a group mean forces this much negative autocorrelation
    assert plain > 0.6, "between-month differences should make neighbouring blocks alike"
    assert abs(within - bias) < 3 / np.sqrt(n), "but nothing is ordered inside a month"


def test_order_inside_a_group_survives_removing_the_group_mean():
    s = _slow_series(30_000)
    within, _ = so.within_group_block_autocorr(s, pd.Series(np.repeat(np.arange(3), 10_000)), 500)
    assert within > 0.3

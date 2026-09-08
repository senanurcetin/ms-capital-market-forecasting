"""Tests for the sequence-shape features.

The premise of the module is a claim about information: aggregates are
permutation-invariant, so anything that depends on ORDER is missing from the 292 existing
features. The first group of tests checks that claim directly on reference
implementations - shuffle the snapshots and show that a mean does not move while path
efficiency and autocorrelation do. If that failed, the whole module would be redundant
with what already exists.

The rest guard the SQL against the two defects already found in it: a non-deterministic
ANY_VALUE, and CORR returning NaN on samples whose mid never moves.
"""
import numpy as np
import pytest

from src.features.shape_features import ORDER, build_sql, feature_names


# ------------------------------------------------- the premise: order carries information

def _path_efficiency(mid):
    d = np.diff(mid)
    return abs(d.sum()) / np.abs(d).sum()


def _autocorr1(mid):
    d = np.diff(mid)
    return float(np.corrcoef(d[1:], d[:-1])[0, 1])


def _same_values_different_order():
    """Two paths visiting the SAME multiset of prices in a different order.

    This is the construction that matters. Every order-free statistic - mean, std, min,
    max, any quantile, any window aggregate in the existing 292 - is identical for both,
    because they are computed from the multiset alone. Only a path quantity can separate
    them.

    Note what does NOT work here, found by this test failing: giving the two paths the
    same multiset of STEPS. Path efficiency is |sum d| / sum|d|, and both of those are
    themselves order-free over the steps, so it cannot distinguish such a pair. The
    feature is order-sensitive with respect to the price series, not the step series.
    """
    monotone = np.array([0.0, 1, 2, 3, 4])
    wandering = np.array([0.0, 2, 1, 3, 4])
    return monotone, wandering


def test_order_free_aggregates_cannot_tell_the_two_paths_apart():
    """The control: every aggregate of the prices is identical for both."""
    a, b = _same_values_different_order()
    assert sorted(a.tolist()) == sorted(b.tolist())
    for stat in (np.mean, np.std, np.min, np.max, np.median):
        assert stat(a) == pytest.approx(stat(b))


def test_path_efficiency_does_tell_them_apart():
    """The whole justification for the module, in one assertion."""
    a, b = _same_values_different_order()
    assert _path_efficiency(a) == pytest.approx(1.0)      # monotone: no wasted motion
    assert _path_efficiency(b) < 0.7                       # wandering: doubles back


def test_shuffling_leaves_the_mean_alone_but_moves_the_shape():
    rng = np.random.default_rng(0)
    mid = 1.0 + np.cumsum(rng.normal(0, 1e-4, 200))
    shuffled = mid.copy()
    rng.shuffle(shuffled)

    assert mid.mean() == pytest.approx(shuffled.mean())
    assert mid.std() == pytest.approx(shuffled.std())
    assert _path_efficiency(mid) != pytest.approx(_path_efficiency(shuffled), rel=1e-3)


def test_autocorrelation_separates_reverting_from_trending():
    """Alternating steps mean-revert (negative ac1); persistent steps trend (positive)."""
    reverting = np.cumsum([1.0, -1, 1, -1, 1, -1, 1, -1, 1, -1])
    trending = np.cumsum([1.0, 1, 1, 1, 1, -1, -1, 1, 1, 1])
    assert _autocorr1(reverting) < 0 < _autocorr1(trending)


def test_path_efficiency_is_bounded():
    rng = np.random.default_rng(1)
    for _ in range(20):
        mid = np.cumsum(rng.normal(0, 1, 50))
        assert 0.0 <= _path_efficiency(mid) <= 1.0 + 1e-12


# ----------------------------------------------------------------- the SQL it generates

def test_every_feature_is_namespaced():
    names = feature_names()
    assert names and all(n.startswith("shp_") for n in names)


def test_feature_names_are_unique():
    names = feature_names()
    assert len(names) == len(set(names)), "a duplicated alias would silently shadow one"


def test_sql_orders_chronologically():
    """seconds_before_predict counts DOWN to the prediction instant.

    Ordering ASC would reverse time and silently flip every slope's sign.
    """
    assert "ORDER BY seconds_before_predict DESC" in ORDER


def test_sql_uses_no_nondeterministic_aggregate():
    """ANY_VALUE picks an arbitrary row, which made two features unreproducible."""
    assert "ANY_VALUE(" not in build_sql("train")


def test_correlations_are_guarded_against_nan():
    """CORR returns NaN when a sample's mid never moves - 5.0% of samples.

    NaN survives into parquet, breaks Ridge's imputation and poisons any mean taken over
    the column, so every CORR must be wrapped.
    """
    sql = build_sql("train")
    assert sql.count("IS_NAN(CORR(") == sql.count("CORR(") // 2


def test_train_and_test_sql_differ_only_in_the_split():
    train, test = build_sql("train"), build_sql("test")
    assert train != test
    assert train.replace("_train", "_SPLIT") == test.replace("_test", "_SPLIT")


def test_no_lookahead_column_is_referenced():
    """Nothing may reach past the prediction instant; only the lookback is available."""
    sql = build_sql("train")
    assert "target" not in sql and "month" not in sql

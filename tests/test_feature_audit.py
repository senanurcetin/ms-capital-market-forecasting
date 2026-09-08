"""Tests for the feature-redundancy audit.

The non-trivial part is the collapse. Counting duplicate PAIRS overstates the removal:
if a == b and b == c, that is three features collapsing to one, not two independent
removals. The union-find has to be transitive or the effective count comes out wrong.
"""
import numpy as np
import pandas as pd

from src.evaluation.feature_audit import (
    NEAR_DUPLICATE, correlation_pairs, effective_count,
)


def _frame(**cols):
    return pd.DataFrame(cols).astype("float32")


def test_a_scaled_copy_is_found_as_a_duplicate():
    """The real case: a rate is a count divided by a constant window length."""
    rng = np.random.default_rng(0)
    n = rng.normal(0, 1, 500)
    df = _frame(count=n, rate=n / 60.0, other=rng.normal(0, 1, 500))
    pairs = correlation_pairs(df, ["count", "rate", "other"])
    top = pairs.iloc[0]
    assert {top.a, top.b} == {"count", "rate"}
    assert top.abs_corr > 0.999


def test_a_shifted_copy_is_also_found():
    """Correlation is invariant to an offset, so an added constant hides nothing."""
    rng = np.random.default_rng(1)
    v = rng.normal(0, 1, 500)
    df = _frame(a=v, b=v + 7.0)
    assert correlation_pairs(df, ["a", "b"]).iloc[0].abs_corr > 0.999


def test_a_negated_copy_counts_as_duplicate():
    """The audit uses ABSOLUTE correlation: a sign flip carries no new information."""
    rng = np.random.default_rng(2)
    v = rng.normal(0, 1, 500)
    df = _frame(a=v, b=-v)
    assert correlation_pairs(df, ["a", "b"]).iloc[0].abs_corr > 0.999


def test_independent_features_are_not_flagged():
    rng = np.random.default_rng(3)
    df = _frame(a=rng.normal(0, 1, 2000), b=rng.normal(0, 1, 2000))
    assert correlation_pairs(df, ["a", "b"]).iloc[0].abs_corr < 0.2


def test_pairs_are_sorted_and_complete():
    rng = np.random.default_rng(4)
    cols = ["a", "b", "c", "d"]
    df = _frame(**{c: rng.normal(0, 1, 300) for c in cols})
    pairs = correlation_pairs(df, cols)
    assert len(pairs) == 4 * 3 // 2, "every unordered pair exactly once"
    assert (pairs.abs_corr.to_numpy()[:-1] >= pairs.abs_corr.to_numpy()[1:]).all()


def test_collapse_is_transitive():
    """a == b and b == c is ONE surviving feature, not two removals from three."""
    pairs = pd.DataFrame({"a": ["a", "b"], "b": ["b", "c"], "abs_corr": [1.0, 1.0]})
    assert effective_count(pairs, ["a", "b", "c"]) == 1


def test_collapse_leaves_distinct_groups_alone():
    pairs = pd.DataFrame({"a": ["a", "c"], "b": ["b", "d"], "abs_corr": [1.0, 1.0]})
    assert effective_count(pairs, ["a", "b", "c", "d", "e"]) == 3


def test_nothing_collapses_below_the_threshold():
    pairs = pd.DataFrame({"a": ["a"], "b": ["b"], "abs_corr": [NEAR_DUPLICATE - 0.01]})
    assert effective_count(pairs, ["a", "b"]) == 2


def test_effective_count_never_exceeds_the_raw_count():
    rng = np.random.default_rng(5)
    cols = [f"f{i}" for i in range(6)]
    df = _frame(**{c: rng.normal(0, 1, 400) for c in cols})
    pairs = correlation_pairs(df, cols)
    assert 1 <= effective_count(pairs, cols) <= len(cols)


def test_nan_and_inf_do_not_poison_a_column():
    """The feature layer emits NaN by design; one infinity must not blank a whole column."""
    rng = np.random.default_rng(6)
    v = rng.normal(0, 1, 300)
    a, b = v.copy(), v.copy()
    a[5] = np.nan
    b[9] = np.inf
    pairs = correlation_pairs(_frame(a=a, b=b), ["a", "b"])
    assert np.isfinite(pairs.abs_corr).all()

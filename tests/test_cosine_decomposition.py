"""Tests for the subgroup decomposition of a pooled cosine.

The decomposition is an exact algebraic identity, not an approximation, so the tests hold
it to floating-point precision rather than a tolerance band. If it ever fails, the
conclusion drawn from it - that cosine weights subgroups by magnitude and not by sample
count - is unsafe.
"""
import numpy as np
import pytest

from src.evaluation.cosine_decomposition import decompose, verify_identity
from src.evaluation.metrics import cosine_similarity


def _data(n=500, k=4, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.normal(0, 0.0026, n)
    pred = 0.15 * y + rng.normal(0, 0.0004, n)     # weak but real skill
    group = rng.integers(0, k, n).astype(str)
    return y, pred, group


def test_identity_is_exact():
    y, pred, g = _data()
    r = verify_identity(y, pred, g)
    assert r["abs_error"] < 1e-15


def test_identity_holds_for_wildly_unequal_group_scales():
    """The case the analysis actually depends on: groups differing in magnitude."""
    rng = np.random.default_rng(1)
    y = np.concatenate([rng.normal(0, 1e-4, 300), rng.normal(0, 1e-2, 300)])
    pred = 0.2 * y + rng.normal(0, 1e-5, 600)
    g = np.array(["small"] * 300 + ["large"] * 300)
    assert verify_identity(y, pred, g)["abs_error"] < 1e-15


def test_identity_holds_for_unbalanced_groups():
    y, pred, _ = _data(n=600)
    g = np.array(["a"] * 590 + ["b"] * 10)
    assert verify_identity(y, pred, g)["abs_error"] < 1e-15


def test_single_group_reduces_to_the_pooled_cosine():
    y, pred, _ = _data()
    parts = decompose(y, pred, np.zeros(len(y)).astype(str))
    assert parts.weight.iloc[0] == pytest.approx(1.0)
    assert parts.cosine.iloc[0] == pytest.approx(cosine_similarity(y, pred))


def test_weights_never_exceed_one():
    """Cauchy-Schwarz: sum ||y_g|| ||p_g|| <= ||y|| ||p||."""
    for seed in range(5):
        y, pred, g = _data(seed=seed)
        assert decompose(y, pred, g).weight.sum() <= 1.0 + 1e-12


def test_weights_sum_to_one_when_groups_are_proportional():
    """Equality in Cauchy-Schwarz: groups whose (y, p) pairs are scaled copies."""
    base_y = np.array([1.0, -2.0, 3.0])
    base_p = np.array([0.5, -1.0, 1.5])
    y = np.concatenate([base_y, 2 * base_y])
    pred = np.concatenate([base_p, 2 * base_p])
    g = np.array(["a"] * 3 + ["b"] * 3)
    assert decompose(y, pred, g).weight.sum() == pytest.approx(1.0)


def test_magnitude_weight_differs_from_count_weight():
    """The whole point: equal-sized groups get UNEQUAL weight when magnitudes differ."""
    rng = np.random.default_rng(2)
    y = np.concatenate([rng.normal(0, 1e-4, 200), rng.normal(0, 1e-2, 200)])
    pred = 0.2 * y + rng.normal(0, 1e-6, 400)
    g = np.array(["quiet"] * 200 + ["loud"] * 200)
    parts = decompose(y, pred, g).set_index("group")
    assert parts.loc["quiet", "count_weight"] == pytest.approx(parts.loc["loud", "count_weight"])
    assert parts.loc["loud", "weight"] > 10 * parts.loc["quiet", "weight"]


def test_count_weighting_can_invert_the_verdict():
    """A count-weighted average is not the metric, and the difference is not cosmetic.

    Note that merely rescaling one group's target does NOT do this - cosine is
    scale-invariant, so both groups would still score ~1. The divergence needs the SKILL
    to differ between a quiet group and a loud one. Here the model is near-perfect on the
    small-magnitude half and useless on the large-magnitude half: counting samples calls
    it skilful, while the metric - which weights by magnitude - correctly calls it worthless.
    """
    rng = np.random.default_rng(3)
    y_quiet, y_loud = rng.normal(0, 1e-4, 200), rng.normal(0, 1e-2, 200)
    pred = np.concatenate([
        y_quiet + rng.normal(0, 1e-5, 200),          # near-perfect where it does not count
        0.02 * y_loud + rng.normal(0, 2e-2, 200),    # useless where it does
    ])
    y = np.concatenate([y_quiet, y_loud])
    g = np.array(["quiet"] * 200 + ["loud"] * 200)

    parts = decompose(y, pred, g)
    by_count = (parts.count_weight * parts.cosine).sum()
    pooled = cosine_similarity(y, pred)

    assert by_count > 0.4, "count weighting reads as skilful"
    assert pooled < 0.0, "the metric reads as no skill at all"


def test_contribution_column_rebuilds_the_pooled_score():
    y, pred, g = _data(seed=7)
    parts = decompose(y, pred, g)
    assert parts.contribution.sum() == pytest.approx(cosine_similarity(y, pred))


def test_group_order_does_not_change_the_result():
    y, pred, g = _data(seed=9)
    a = decompose(y, pred, g)
    idx = np.random.default_rng(0).permutation(len(y))
    b = decompose(y[idx], pred[idx], g[idx])
    assert a.set_index("group").cosine.round(12).to_dict() == \
           b.set_index("group").cosine.round(12).to_dict()

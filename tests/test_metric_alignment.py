"""Tests for the metric-alignment experiment.

The experiment failed, and decisively - which makes the mechanics worth pinning down, so
that "weighting by |y| hurts" is a statement about the idea rather than about a bug in how
the weights were built.
"""
import numpy as np
import pytest
from src.models.metric_alignment import ALPHAS, weights


def test_alpha_zero_is_the_untouched_code_path():
    """None, not a vector of ones.

    A control arm that is bit-identical to production is a better control than one that
    merely ought to be equivalent.
    """
    assert weights(np.array([0.1, -0.2, 0.3]), 0.0) is None


def test_weights_are_normalised_to_mean_one():
    """Otherwise alpha would change the effective learning rate as well as the emphasis,
    and the comparison would confound two things."""
    rng = np.random.default_rng(0)
    y = rng.normal(0, 0.0026, 1000)
    for a in (0.5, 1.0):
        assert weights(y, a).mean() == pytest.approx(1.0)


def test_larger_targets_get_more_weight():
    w = weights(np.array([0.001, 0.002, 0.004]), 1.0)
    assert w[0] < w[1] < w[2]


def test_zero_targets_are_floored_not_dropped():
    """5.5% of targets are exactly zero. At alpha >= 1 they would get zero weight and
    leave training entirely - which is a different experiment from reweighting, and would
    discard 70k rows that still say where the boundary between moving and not moving is.
    """
    y = np.array([0.0, 0.0, 0.001, 0.005])
    for a in (0.5, 1.0):
        assert (weights(y, a) > 0).all()


def test_the_floor_does_not_swamp_the_signal():
    """Flooring must not flatten the weights into near-uniformity, or the arms would be
    indistinguishable and a null would mean nothing."""
    rng = np.random.default_rng(1)
    y = rng.normal(0, 0.0026, 5000)
    w = weights(y, 1.0)
    assert w.max() / w.min() > 5


def test_stronger_alpha_concentrates_weight_further():
    rng = np.random.default_rng(2)
    y = rng.normal(0, 0.0026, 5000)
    half, full = weights(y, 0.5), weights(y, 1.0)
    assert full.std() > half.std()


def test_weights_depend_only_on_the_magnitude():
    """Sign carries direction, not importance; +y and -y must weigh the same."""
    w = weights(np.array([0.003, -0.003, 0.001]), 1.0)
    assert w[0] == pytest.approx(w[1])


def test_alphas_include_the_control():
    assert 0.0 in ALPHAS, "without an unweighted arm there is nothing to compare against"


def test_weights_are_finite_for_a_degenerate_target():
    """All-zero targets would make the floor undefined if it were taken over everything."""
    y = np.array([0.0, 0.0, 0.0, 1e-6])
    w = weights(y, 1.0)
    assert np.isfinite(w).all() and (w > 0).all()

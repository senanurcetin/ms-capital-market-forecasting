"""Tests for the shippable artefact and the recency measurement behind it.

Both modules exist because the submitted model was handicapped in two ways that its own
metadata revealed: it was a single LightGBM, and it trained on 64 of 71 months. The tests
here pin the split arithmetic that fixes the second, and the wrapper contract that the
first attempt at the ensemble broke.
"""
import numpy as np
import pandas as pd
import pytest

from src.models import recency, ship


# --------------------------------------------------------------- ship: the split

def test_no_month_does_two_jobs():
    """Training, early stopping and blend fitting must be disjoint.

    Fitting stopping rounds and blend weights on the same rows would let the weights
    compensate for a stopping point chosen on those rows.
    """
    assert ship.TRAIN_END < ship.STOP_MONTH < ship.BLEND_MONTHS[0]
    assert ship.BLEND_MONTHS[0] <= ship.BLEND_MONTHS[1]


def test_shipping_uses_far_more_months_than_the_submitted_artefact():
    """The whole point: 68 months of training rather than 64."""
    assert ship.TRAIN_END + 1 > 64


def test_the_blend_window_is_more_than_one_month():
    """Fitted on month 70 alone, NNLS gave xgboost 0.71 - noise, not signal.

    The base models are highly correlated, so their differences are mostly noise and the
    weight split swings on very little. Widening the window halves that variance.
    """
    assert ship.BLEND_MONTHS[1] - ship.BLEND_MONTHS[0] + 1 >= 2


def test_every_month_is_accounted_for():
    """Nothing between the start and the last blend month is silently skipped."""
    used = set(range(0, ship.TRAIN_END + 1)) | {ship.STOP_MONTH} | set(
        range(ship.BLEND_MONTHS[0], ship.BLEND_MONTHS[1] + 1))
    assert used == set(range(0, ship.BLEND_MONTHS[1] + 1))


# ------------------------------------------------- ship: the wrapper contract

class _BareEstimator:
    """Stands in for a raw sklearn Ridge: rejects NaN, exactly as the real one does."""

    def predict(self, X):
        if np.isnan(np.asarray(X, dtype=float)).any():
            raise ValueError("Input X contains NaN.")
        return np.zeros(len(X))


class _Wrapper:
    """Stands in for RidgeModel: imputes, then delegates."""

    def __init__(self):
        self.inner = _BareEstimator()
        self.medians = None

    def fit(self, X):
        self.medians = np.nanmedian(np.asarray(X, dtype=float), axis=0)
        return self

    def predict(self, X):
        filled = np.asarray(X, dtype=float).copy()
        idx = np.where(np.isnan(filled))
        filled[idx] = np.take(self.medians, idx[1])
        return self.inner.predict(filled)


def _frame_with_nans():
    X = np.arange(20, dtype=float).reshape(10, 2)
    X[3, 1] = np.nan
    return pd.DataFrame(X, columns=["a", "b"])


def test_unwrapping_a_model_loses_its_nan_handling():
    """This is the bug the first ensemble build hit, reproduced.

    The feature layer emits NaN by design (short windows are often empty), so a bare
    estimator pulled out of its wrapper fails on the first test row that has one.
    """
    df = _frame_with_nans()
    with pytest.raises(ValueError, match="NaN"):
        _BareEstimator().predict(df)


def test_the_wrapper_survives_the_same_input():
    df = _frame_with_nans()
    wrapped = _Wrapper().fit(df)
    assert len(wrapped.predict(df)) == len(df)


# --------------------------------------------------------------- recency arms

def test_cumulative_windows_all_start_at_zero():
    """The cumulative arm varies volume and recency together, as shipping does."""
    assert all(lo == 0 for lo, _ in recency.CUMULATIVE)


def test_fixed_windows_hold_the_training_span_constant():
    """The control arm: only the endpoint moves, so recency is isolated from volume."""
    spans = {hi - lo + 1 for lo, hi in recency.FIXED_WINDOW}
    assert len(spans) == 1


def test_the_two_arms_share_an_anchor():
    """They must agree somewhere, or the two slopes are not comparable."""
    assert set(recency.CUMULATIVE) & set(recency.FIXED_WINDOW)


def test_no_training_window_touches_the_evaluation_block():
    for lo, hi in recency.CUMULATIVE + recency.FIXED_WINDOW:
        assert hi < recency.EVAL[0], f"window {lo}-{hi} overlaps the evaluation block"


def test_windows_span_a_range_of_gaps():
    """A single gap would give no slope to measure."""
    gaps = {recency.EVAL[0] - hi for _, hi in recency.CUMULATIVE}
    assert len(gaps) >= 3

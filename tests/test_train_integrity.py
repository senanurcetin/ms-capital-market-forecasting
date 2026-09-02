"""Does the training harness's leakage guard actually work?

These tests do not merely check that a correct setup passes; they INJECT broken
setups and prove the guard catches them.
"""
import numpy as np
import pytest

from src.evaluation.temporal_validation import Fold, build_folds, holdout_months
from src.models.train import assert_fold_integrity


@pytest.fixture
def months():
    return np.repeat(np.arange(71), 50)


def _idx(months, lo, hi):
    return np.flatnonzero((months >= lo) & (months <= hi))


def test_config_folds_all_pass(months):
    for fold in build_folds():
        tr = _idx(months, *fold.train_months)
        va = _idx(months, *fold.val_months)
        assert_fold_integrity(months, fold, tr, va)


def test_catches_train_after_val(months):
    """Train placed AFTER validation must be caught."""
    bad = Fold(index=99, train_months=(40, 50), val_months=(10, 20), embargo_months=(21, 39))
    with pytest.raises(AssertionError, match="not strictly before"):
        assert_fold_integrity(months, bad, _idx(months, 40, 50), _idx(months, 10, 20))


def test_catches_embargo_violation(months):
    """Using data from an embargo month must be caught."""
    fold = build_folds()[0]
    gap_lo = fold.embargo_months[0]
    tr = np.concatenate([_idx(months, *fold.train_months), _idx(months, gap_lo, gap_lo)])
    with pytest.raises(AssertionError, match="embargo month"):
        assert_fold_integrity(months, fold, tr, _idx(months, *fold.val_months))


def test_catches_holdout_leak(months):
    """Hold-out months leaking in must be caught.

    The violation is injected on the VALIDATION side: adding it to train would
    trigger the "not strictly before" rule first and the hold-out rule could not
    be tested in isolation.
    """
    ho_lo, _ = holdout_months()
    fold = build_folds()[-1]                      # val 60-64, hold-out 65-70
    va = np.concatenate([_idx(months, *fold.val_months), _idx(months, ho_lo, ho_lo)])
    with pytest.raises(AssertionError, match="hold-out"):
        assert_fold_integrity(months, fold, _idx(months, *fold.train_months), va)


def test_train_side_holdout_leak_also_caught(months):
    """Hold-out leaking into train must also be caught (by a different rule)."""
    ho_lo, _ = holdout_months()
    fold = build_folds()[0]
    tr = np.concatenate([_idx(months, *fold.train_months), _idx(months, ho_lo, ho_lo)])
    with pytest.raises(AssertionError):
        assert_fold_integrity(months, fold, tr, _idx(months, *fold.val_months))


def test_catches_overlapping_indices(months):
    """The same row appearing in both train and validation must be caught."""
    fold = build_folds()[0]
    va = _idx(months, *fold.val_months)
    tr = np.concatenate([_idx(months, *fold.train_months), va[:5]])
    with pytest.raises(AssertionError):
        assert_fold_integrity(months, fold, tr, va)


def test_random_split_would_be_caught(months):
    """The plan's single biggest risk: a random split. The guard must reject it."""
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(months))
    tr, va = perm[: int(0.8 * len(perm))], perm[int(0.8 * len(perm)) :]
    with pytest.raises(AssertionError):
        assert_fold_integrity(months, build_folds()[0], tr, va)

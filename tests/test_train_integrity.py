"""Egitim harness'inin leakage korumasi gercekten calisiyor mu?

Bu testler sadece 'dogru kurulumda gecmeli' demiyor; BOZUK kurulumlari
enjekte edip korumanin bunlari YAKALADIGINI da kanitliyor.
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
    """Train, val'den SONRA olursa yakalanmali."""
    bad = Fold(index=99, train_months=(40, 50), val_months=(10, 20), embargo_months=(21, 39))
    with pytest.raises(AssertionError, match="train, val'den once degil"):
        assert_fold_integrity(months, bad, _idx(months, 40, 50), _idx(months, 10, 20))


def test_catches_embargo_violation(months):
    """Embargo ayindan veri kullanilirsa yakalanmali."""
    fold = build_folds()[0]
    gap_lo = fold.embargo_months[0]
    tr = np.concatenate([_idx(months, *fold.train_months), _idx(months, gap_lo, gap_lo)])
    with pytest.raises(AssertionError, match="embargo ayi"):
        assert_fold_integrity(months, fold, tr, _idx(months, *fold.val_months))


def test_catches_holdout_leak(months):
    """Hold-out aylari sizarsa yakalanmali.

    Ihlal VAL tarafina enjekte edilir: train'e eklenirse once
    'train, val'den once degil' kurali tetiklenir ve hold-out kurali
    izole test edilemez.
    """
    ho_lo, _ = holdout_months()
    fold = build_folds()[-1]                      # val 60-64, hold-out 65-70
    va = np.concatenate([_idx(months, *fold.val_months), _idx(months, ho_lo, ho_lo)])
    with pytest.raises(AssertionError, match="hold-out"):
        assert_fold_integrity(months, fold, _idx(months, *fold.train_months), va)


def test_train_side_holdout_leak_also_caught(months):
    """Hold-out train'e sizarsa da (baska bir kuralla) yakalanmali."""
    ho_lo, _ = holdout_months()
    fold = build_folds()[0]
    tr = np.concatenate([_idx(months, *fold.train_months), _idx(months, ho_lo, ho_lo)])
    with pytest.raises(AssertionError):
        assert_fold_integrity(months, fold, tr, _idx(months, *fold.val_months))


def test_catches_overlapping_indices(months):
    """Ayni satir hem train hem val'deyse yakalanmali."""
    fold = build_folds()[0]
    va = _idx(months, *fold.val_months)
    tr = np.concatenate([_idx(months, *fold.train_months), va[:5]])
    with pytest.raises(AssertionError):
        assert_fold_integrity(months, fold, tr, va)


def test_random_split_would_be_caught(months):
    """Plandaki en kritik risk: random split. Koruma bunu gecirmemeli."""
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(months))
    tr, va = perm[: int(0.8 * len(perm))], perm[int(0.8 * len(perm)) :]
    with pytest.raises(AssertionError):
        assert_fold_integrity(months, build_folds()[0], tr, va)

"""Temporal split testleri - leakage'in onlendigini kanitlar."""
import numpy as np
import pytest

from src.evaluation.temporal_validation import (
    assert_no_overlap, build_folds, holdout_indices, holdout_months,
    iter_folds, stability,
)


@pytest.fixture
def months():
    """71 ay x 100 sample, ay basina esit dagilim."""
    return np.repeat(np.arange(71), 100)


def test_folds_built_from_config():
    folds = build_folds()
    assert len(folds) == 5
    assert folds[0].train_months == (0, 34)
    assert folds[-1].val_months == (60, 64)


def test_every_fold_has_embargo_gap():
    for f in build_folds():
        gap_lo, gap_hi = f.embargo_months
        assert gap_hi >= gap_lo, f"{f.describe()} embargo bosluğu yok"
        assert f.train_months[1] < gap_lo <= gap_hi < f.val_months[0]


def test_train_always_strictly_before_val(months):
    for fold, tr, va in iter_folds(months):
        assert months[tr].max() < months[va].min(), fold.describe()


def test_no_index_appears_in_both_train_and_val(months):
    for fold, tr, va in iter_folds(months):
        assert not set(tr) & set(va), fold.describe()


def test_holdout_never_touched_by_any_fold(months):
    ho = set(holdout_indices(months))
    assert ho
    for fold, tr, va in iter_folds(months):
        assert not ho & set(tr), f"{fold.describe()} hold-out'u egitimde kullaniyor"
        assert not ho & set(va), f"{fold.describe()} hold-out'u validasyonda kullaniyor"


def test_embargo_months_excluded_from_both_sides(months):
    for fold, tr, va in iter_folds(months):
        used = set(months[tr]) | set(months[va])
        for m in range(fold.embargo_months[0], fold.embargo_months[1] + 1):
            assert m not in used, f"{fold.describe()} embargo ayi {m} kullanilmis"


def test_expanding_window_grows(months):
    sizes = [len(tr) for _, tr, _ in iter_folds(months)]
    assert sizes == sorted(sizes) and sizes[0] < sizes[-1]


def test_assert_no_overlap_passes_on_config(months):
    assert_no_overlap(months)   # exception atmamali


def test_holdout_range_matches_config(months):
    lo, hi = holdout_months()
    assert (lo, hi) == (65, 70)
    assert set(months[holdout_indices(months)]) == set(range(lo, hi + 1))


def test_stability_reports_worst_fold():
    s = stability([0.10, 0.05, 0.12, 0.11, 0.09])
    assert s["worst_fold"] == 2
    assert s["min"] == pytest.approx(0.05)
    assert s["std"] > 0

"""Temporal split tests - proof that leakage is prevented."""
import numpy as np
import pytest
from src.evaluation.temporal_validation import (
    assert_no_overlap,
    build_folds,
    holdout_indices,
    holdout_months,
    iter_folds,
    split_indices,
    stability,
)


@pytest.fixture
def months():
    """71 months x 100 samples, evenly distributed."""
    return np.repeat(np.arange(71), 100)


def test_folds_built_from_config():
    folds = build_folds()
    assert len(folds) == 5
    assert folds[0].train_months == (0, 34)
    assert folds[-1].val_months == (60, 64)


def test_every_fold_has_embargo_gap():
    for f in build_folds():
        gap_lo, gap_hi = f.embargo_months
        assert gap_hi >= gap_lo, f"{f.describe()} has no embargo gap"
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
        assert not ho & set(tr), f"{fold.describe()} trains on hold-out data"
        assert not ho & set(va), f"{fold.describe()} validates on hold-out data"


def test_embargo_months_excluded_from_both_sides(months):
    for fold, tr, va in iter_folds(months):
        used = set(months[tr]) | set(months[va])
        for m in range(fold.embargo_months[0], fold.embargo_months[1] + 1):
            assert m not in used, f"{fold.describe()} used embargo month {m}"


def test_expanding_window_grows(months):
    sizes = [len(tr) for _, tr, _ in iter_folds(months)]
    assert sizes == sorted(sizes) and sizes[0] < sizes[-1]


def test_assert_no_overlap_passes_on_config(months):
    assert_no_overlap(months)   # must not raise


def test_holdout_range_matches_config(months):
    lo, hi = holdout_months()
    assert (lo, hi) == (65, 70)
    assert set(months[holdout_indices(months)]) == set(range(lo, hi + 1))


def test_stability_reports_worst_fold():
    s = stability([0.10, 0.05, 0.12, 0.11, 0.09])
    assert s["worst_fold"] == 2
    assert s["min"] == pytest.approx(0.05)
    assert s["std"] > 0

def test_assert_no_overlap_checks_the_supplied_month_vector():
    for invalid in (
        np.array([]),
        np.array([64.0, np.nan, 66.0]),
        np.array([-1, 0, 65]),
        np.array([63.0, 63.5, 65.0]),
    ):
        with pytest.raises(ValueError):
            assert_no_overlap(invalid)


def test_holdout_only_rejects_pre_holdout_and_unordered_months():
    from src.evaluation.temporal_validation import assert_holdout_only

    assert_holdout_only(np.array([65, 65, 66, 70]))
    with pytest.raises(ValueError, match="outside hold-out"):
        assert_holdout_only(np.array([64, 65, 70]))
    with pytest.raises(ValueError, match="chronological"):
        assert_holdout_only(np.array([66, 65, 70]))
    with pytest.raises(ValueError, match="nonempty"):
        assert_holdout_only(np.array([]))


@pytest.mark.parametrize("scores", [
    [],
    [float("nan")],
    [0.1, float("inf")],
    [float("-inf"), 0.1],
])
def test_stability_rejects_empty_or_nonfinite_fold_evidence(scores):
    with pytest.raises(ValueError, match="nonempty 1-D finite"):
        stability(scores)


def test_stability_accepts_one_finite_fold_without_spurious_sample_std():
    report = stability([-0.02])
    assert report["mean"] == pytest.approx(-0.02)
    assert report["std"] == 0.0
    assert report["worst_fold"] == 1



@pytest.mark.parametrize("invalid", [
    np.array([]),
    np.array([64, np.nan]),
    np.array([-1, 65]),
    np.array([64.5, 65.0]),
    np.array([[64, 65]]),
    np.array([True, False]),
])
def test_public_temporal_split_helpers_reject_invalid_month_labels(invalid):
    fold = build_folds()[0]
    with pytest.raises(ValueError, match="months must"):
        split_indices(invalid, fold)
    with pytest.raises(ValueError, match="months must"):
        holdout_indices(invalid)
    with pytest.raises(ValueError, match="months must"):
        list(iter_folds(invalid))


def test_public_temporal_helpers_keep_valid_repeated_month_rows(months):
    fold = build_folds()[0]
    tr, val = split_indices(months, fold)
    assert len(tr) > 0 and len(val) > 0
    assert set(tr).isdisjoint(val)
    assert np.array_equal(months[holdout_indices(months)], np.repeat(np.arange(65, 71), 100))


@pytest.mark.parametrize(("folds", "embargo", "holdout", "message"), [
    ([{"train_end": 34, "val_start": 36, "val_end": 35}], 1,
     {"start": 65, "end": 70}, "validation end"),
    ([{"train_end": 34, "val_start": 36, "val_end": 65}], 1,
     {"start": 65, "end": 70}, "overlaps hold-out"),
    ([{"train_end": 34.5, "val_start": 36, "val_end": 40}], 1,
     {"start": 65, "end": 70}, "nonnegative integers"),
    ([{"train_end": 34, "val_start": 36, "val_end": 40},
      {"train_end": 35, "val_start": 39, "val_end": 45}], 1,
     {"start": 65, "end": 70}, "chronologically"),
    ([{"train_end": 34, "val_start": 36, "val_end": 40}], 0,
     {"start": 65, "end": 70}, "positive integer"),
    ([], 1, {"start": 65, "end": 70}, "at least one fold"),
])
def test_invalid_walkforward_configuration_fails_before_partitioning(
    monkeypatch, folds, embargo, holdout, message
):
    from types import SimpleNamespace

    import src.evaluation.temporal_validation as temporal
    monkeypatch.setattr(temporal, "load_config", lambda: SimpleNamespace(
        validation=SimpleNamespace(
            embargo_months=embargo, folds=folds, holdout=holdout
        )
    ))
    with pytest.raises(ValueError, match=message):
        temporal.build_folds()


@pytest.mark.parametrize("holdout", [
    {"start": 70, "end": 65},
    {"start": -1, "end": 70},
    {"start": 65.5, "end": 70},
])
def test_invalid_holdout_bounds_are_rejected(monkeypatch, holdout):
    from types import SimpleNamespace

    import src.evaluation.temporal_validation as temporal
    monkeypatch.setattr(temporal, "load_config", lambda: SimpleNamespace(
        validation=SimpleNamespace(holdout=holdout)
    ))
    with pytest.raises(ValueError, match="ordered nonnegative"):
        temporal.holdout_months()


@pytest.mark.parametrize("run", [
    lambda months: list(iter_folds(months)),
    assert_no_overlap,
])
def test_walkforward_rejects_out_of_order_month_rows(run, months):
    scrambled = months.copy()
    scrambled[200:300] = 4
    with pytest.raises(ValueError, match="chronological"):
        run(scrambled)


@pytest.mark.parametrize("run", [
    lambda months: list(iter_folds(months)),
    assert_no_overlap,
])
def test_walkforward_rejects_months_beyond_configured_holdout_end(run, months):
    future = np.r_[months, [71]]
    with pytest.raises(ValueError, match="configured hold-out end"):
        run(future)


def test_walkforward_still_accepts_train_only_chronological_rows(months):
    for fold, train, val in iter_folds(months[months <= 64]):
        assert (months[months <= 64][train] < fold.val_months[0]).all()
        assert len(val) > 0

"""Tests for the drift-robustness experiment.

The experiment's conclusion rests entirely on its design being sound, so what is worth
testing is the design, not the arithmetic: that pruning selects on the right side of the
threshold, that the trend statistic reports what it claims, and that the fixed training
window really is fixed - a sliding one would confound gap with training-set size and
invent a trend out of nothing.
"""
import numpy as np
import pandas as pd
import pytest

from src.evaluation.drift_robustness import (
    EVAL_BLOCKS, TRAIN_MONTHS, pruned_columns, trend,
)


def _drift():
    return pd.DataFrame({
        "feature": ["a", "b", "c", "d"],
        "shift": [0.40, -0.25, 0.05, -0.01],
        "abs_shift": [0.40, 0.25, 0.05, 0.01],
    })


def test_pruning_drops_at_or_above_the_threshold():
    keep = pruned_columns(["a", "b", "c", "d"], _drift(), threshold=0.2)
    assert keep == ["c", "d"]


def test_pruning_is_inclusive_at_the_boundary():
    """A feature exactly at the threshold is dropped, matching '|shift| >= threshold'."""
    d = pd.DataFrame({"feature": ["x"], "shift": [0.2], "abs_shift": [0.2]})
    assert pruned_columns(["x"], d, threshold=0.2) == []


def test_pruning_preserves_column_order():
    """Order must follow the dataframe, not the drift ranking - LightGBM is positional."""
    keep = pruned_columns(["d", "c", "b", "a"], _drift(), threshold=0.2)
    assert keep == ["d", "c"]


def test_pruning_ignores_columns_absent_from_the_report():
    """A feature with no drift row is kept, not silently dropped."""
    keep = pruned_columns(["a", "unknown"], _drift(), threshold=0.2)
    assert keep == ["unknown"]


def test_a_looser_threshold_keeps_a_superset():
    cols = ["a", "b", "c", "d"]
    strict = set(pruned_columns(cols, _drift(), threshold=0.1))
    loose = set(pruned_columns(cols, _drift(), threshold=0.3))
    assert strict.issubset(loose)


def test_trend_recovers_a_planted_slope():
    gaps = np.array([b[0] - TRAIN_MONTHS[1] for b in EVAL_BLOCKS], dtype=float)
    out = pd.DataFrame({"gap_months": gaps, "lift": 0.001 * gaps + 0.05})
    slope, corr = trend(out)
    assert slope == pytest.approx(0.001)
    assert corr > 0.999


def test_trend_reports_no_correlation_for_flat_lift():
    gaps = np.array([b[0] - TRAIN_MONTHS[1] for b in EVAL_BLOCKS], dtype=float)
    rng = np.random.default_rng(0)
    out = pd.DataFrame({"gap_months": gaps, "lift": rng.normal(0, 1e-6, len(gaps))})
    slope, _ = trend(out)
    assert abs(slope) < 1e-6


def test_evaluation_blocks_start_after_training_and_never_overlap_it():
    """The confound this design exists to avoid: evaluating on trained months."""
    assert all(lo > TRAIN_MONTHS[1] for lo, _ in EVAL_BLOCKS)


def test_gaps_are_increasing_and_distinct():
    gaps = [lo - TRAIN_MONTHS[1] for lo, _ in EVAL_BLOCKS]
    assert gaps == sorted(gaps) and len(set(gaps)) == len(gaps)


def test_blocks_do_not_overlap_each_other():
    for (_, hi), (lo, _) in zip(EVAL_BLOCKS, EVAL_BLOCKS[1:]):
        assert lo > hi, "overlapping blocks would correlate the lift measurements"

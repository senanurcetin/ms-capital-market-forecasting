"""Temporal (walk-forward) validation.

WHY RANDOM SPLITS ARE FORBIDDEN:
    Consecutive samples can have overlapping lookback windows. A random split
    scatters those near-duplicate rows across folds and inflates the score.
    The only legitimate time axis is label.month (0..70).

EMBARGO:
    embargo_months are dropped entirely between the train and validation ranges,
    which cuts leakage from overlapping windows at the boundary
    (purged/embargoed CV, Lopez de Prado).

HOLD-OUT:
    Months 65-70 are never used for tuning; they are measured ONCE after model
    selection is complete.
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from src.config import load_config


@dataclass(frozen=True)
class Fold:
    """One expanding-window split, with the embargo made explicit.

    The embargo months belong to NEITHER side. Consecutive samples can share overlapping
    lookback windows, so training right up to the validation boundary would let the same
    market minutes appear on both sides of the split. Frozen, because a fold that can be
    mutated after the integrity check has been verified is not verified.
    """

    index: int
    train_months: tuple[int, int]      # (inclusive, inclusive)
    val_months: tuple[int, int]        # (inclusive, inclusive)
    embargo_months: tuple[int, int]    # (inclusive, inclusive) - used by neither side

    def describe(self) -> str:
        """One line naming all three month ranges, for logs and fold-integrity errors."""
        return (
            f"Fold {self.index}: train months {self.train_months[0]}-{self.train_months[1]} "
            f"| embargo {self.embargo_months[0]}-{self.embargo_months[1]} "
            f"| val months {self.val_months[0]}-{self.val_months[1]}"
        )


def build_folds() -> list[Fold]:
    """Turn the expanding-window fold definitions in config.yaml into Fold objects."""
    cfg = load_config()
    embargo = cfg.validation.embargo_months
    folds: list[Fold] = []
    for i, f in enumerate(cfg.validation.folds, start=1):
        train_end = f["train_end"]
        val_start = f["val_start"]
        gap = (train_end + 1, val_start - 1)
        if val_start - train_end - 1 < embargo:
            raise ValueError(
                f"Fold {i}: need at least {embargo} embargo month(s) between "
                f"train_end={train_end} and val_start={val_start}"
            )
        folds.append(
            Fold(
                index=i,
                train_months=(0, train_end),
                val_months=(val_start, f["val_end"]),
                embargo_months=gap,
            )
        )
    return folds


def holdout_months() -> tuple[int, int]:
    """The untouched final months, as (first, last) inclusive.

    Read once, at the very end. Notebook 05 shows this period is an unusually FAVOURABLE
    draw - 83rd percentile of difficulty - so the walk-forward mean, not this, is the
    honest estimate of what a fresh period will yield.
    """
    h = load_config().validation.holdout
    return h["start"], h["end"]


def split_indices(months: np.ndarray, fold: Fold) -> tuple[np.ndarray, np.ndarray]:
    """Derive train/validation indices from a (n_samples,) month vector."""
    months = _validated_months(months)
    tr = np.flatnonzero((months >= fold.train_months[0]) & (months <= fold.train_months[1]))
    va = np.flatnonzero((months >= fold.val_months[0]) & (months <= fold.val_months[1]))
    return tr, va


def iter_folds(months: np.ndarray) -> Iterator[tuple[Fold, np.ndarray, np.ndarray]]:
    """Yield (fold, train_idx, val_idx) for each split, in chronological order."""
    validated = _validated_months(months)
    for fold in build_folds():
        tr, va = split_indices(validated, fold)
        yield fold, tr, va


def holdout_indices(months: np.ndarray) -> np.ndarray:
    """Row positions falling inside the hold-out months."""
    lo, hi = holdout_months()
    months = _validated_months(months)
    return np.flatnonzero((months >= lo) & (months <= hi))


def _validated_months(months: np.ndarray) -> np.ndarray:
    observed = np.asarray(months)
    if observed.ndim != 1 or observed.size == 0:
        raise ValueError("months must be a nonempty 1-D vector")
    if not np.issubdtype(observed.dtype, np.number) or not np.isfinite(observed).all():
        raise ValueError("months must contain only finite numeric month labels")
    if (observed < 0).any() or (observed != np.floor(observed)).any():
        raise ValueError("months must be nonnegative whole-number labels")
    return observed


def assert_holdout_only(months: np.ndarray) -> None:
    """Validate ordered hold-out observations; repeated rows per month are valid."""
    observed = _validated_months(months)
    lo, hi = holdout_months()
    if ((observed < lo) | (observed > hi)).any():
        raise ValueError(f"evaluation includes data outside hold-out months {lo}-{hi}")
    if (np.diff(observed) < 0).any():
        raise ValueError("hold-out month labels must be chronological")


def assert_no_overlap(months: np.ndarray) -> None:
    """Verify configured ranges and the supplied month-vector partitions."""
    observed = _validated_months(months)
    lo, hi = holdout_months()
    holdout = set(holdout_indices(observed))
    for fold in build_folds():
        tr_lo, tr_hi = fold.train_months
        va_lo, va_hi = fold.val_months
        if tr_hi >= va_lo:
            raise AssertionError(f"{fold.describe()}: train overlaps validation")
        if tr_hi >= lo:
            raise AssertionError(f"{fold.describe()}: train reaches into hold-out ({lo}-{hi})")
        if va_hi >= lo:
            raise AssertionError(f"{fold.describe()}: validation reaches into hold-out ({lo}-{hi})")
        train_idx, validation_idx = split_indices(observed, fold)
        if set(train_idx) & set(validation_idx):
            raise AssertionError(f"{fold.describe()}: sample index overlaps train/validation")
        if (set(train_idx) | set(validation_idx)) & holdout:
            raise AssertionError(f"{fold.describe()}: hold-out sample included in fold")


def stability(scores: list[float]) -> dict[str, float]:
    """Across-fold stability. Monthly target volatility swings by 2.69x, which makes
    this as important a selection criterion as the mean."""
    arr = np.asarray(scores, dtype=float)
    # An empty or nonfinite fold set would otherwise select a model using
    # NaN summary metrics or crash while determining the worst fold.
    if arr.ndim != 1 or arr.size == 0 or not np.isfinite(arr).all():
        raise ValueError("fold scores must be a nonempty 1-D finite sequence")
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "worst_fold": int(np.argmin(arr)) + 1,
    }

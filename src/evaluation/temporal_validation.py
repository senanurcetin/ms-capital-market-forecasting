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

from dataclasses import dataclass
from typing import Iterator

import numpy as np

from src.config import load_config


@dataclass(frozen=True)
class Fold:
    index: int
    train_months: tuple[int, int]      # (inclusive, inclusive)
    val_months: tuple[int, int]        # (inclusive, inclusive)
    embargo_months: tuple[int, int]    # (inclusive, inclusive) - used by neither side

    def describe(self) -> str:
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
    h = load_config().validation.holdout
    return h["start"], h["end"]


def split_indices(months: np.ndarray, fold: Fold) -> tuple[np.ndarray, np.ndarray]:
    """Derive train/validation indices from a (n_samples,) month vector."""
    months = np.asarray(months)
    tr = np.flatnonzero((months >= fold.train_months[0]) & (months <= fold.train_months[1]))
    va = np.flatnonzero((months >= fold.val_months[0]) & (months <= fold.val_months[1]))
    return tr, va


def iter_folds(months: np.ndarray) -> Iterator[tuple[Fold, np.ndarray, np.ndarray]]:
    for fold in build_folds():
        tr, va = split_indices(months, fold)
        yield fold, tr, va


def holdout_indices(months: np.ndarray) -> np.ndarray:
    lo, hi = holdout_months()
    months = np.asarray(months)
    return np.flatnonzero((months >= lo) & (months <= hi))


def assert_no_overlap(months: np.ndarray) -> None:
    """Safety net: no fold may overlap train with validation, and hold-out months
    must never appear in any fold."""
    lo, hi = holdout_months()
    for fold in build_folds():
        tr_lo, tr_hi = fold.train_months
        va_lo, va_hi = fold.val_months
        if tr_hi >= va_lo:
            raise AssertionError(f"{fold.describe()}: train overlaps validation")
        if tr_hi >= lo:
            raise AssertionError(f"{fold.describe()}: train reaches into hold-out ({lo}-{hi})")
        if va_hi >= lo:
            raise AssertionError(f"{fold.describe()}: validation reaches into hold-out ({lo}-{hi})")


def stability(scores: list[float]) -> dict[str, float]:
    """Across-fold stability. Monthly target volatility swings by 2.69x, which makes
    this as important a selection criterion as the mean."""
    arr = np.asarray(scores, dtype=float)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "worst_fold": int(np.argmin(arr)) + 1,
    }

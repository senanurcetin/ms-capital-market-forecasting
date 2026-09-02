"""Temporal (walk-forward) validation.

NEDEN RANDOM SPLIT YASAK:
    Ardisik sample'larin 60 saniyelik pencereleri kesisebilir. Random split
    bu neredeyse-kopya satirlari farkli fold'lara dagitir ve skoru sisirir.
    Tek mesru zaman ekseni label.month (0..70).

EMBARGO:
    train ve validation araligi arasinda embargo_months kadar ay tamamen
    disarida birakilir; bu, sinirdaki ortusen pencerelerin sizmasini keser
    (Lopez de Prado, purged/embargoed CV).

HOLD-OUT:
    Ay 65-70 hicbir tuning'de kullanilmaz; secim bittikten sonra BIR KEZ olculur.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np

from src.config import load_config


@dataclass(frozen=True)
class Fold:
    index: int
    train_months: tuple[int, int]      # (dahil, dahil)
    val_months: tuple[int, int]        # (dahil, dahil)
    embargo_months: tuple[int, int]    # (dahil, dahil) - hicbir tarafta kullanilmaz

    def describe(self) -> str:
        return (
            f"Fold {self.index}: train ay {self.train_months[0]}-{self.train_months[1]} "
            f"| embargo {self.embargo_months[0]}-{self.embargo_months[1]} "
            f"| val ay {self.val_months[0]}-{self.val_months[1]}"
        )


def build_folds() -> list[Fold]:
    """config.yaml'deki genisleyen-pencere fold tanimlarini Fold nesnelerine cevirir."""
    cfg = load_config()
    embargo = cfg.validation.embargo_months
    folds: list[Fold] = []
    for i, f in enumerate(cfg.validation.folds, start=1):
        train_end = f["train_end"]
        val_start = f["val_start"]
        gap = (train_end + 1, val_start - 1)
        if val_start - train_end - 1 < embargo:
            raise ValueError(
                f"Fold {i}: train_end={train_end} ile val_start={val_start} arasinda "
                f"en az {embargo} ay embargo olmali"
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
    """Ay vektorunden (n_samples,) train/val indekslerini uretir."""
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
    """Guvenlik agi: hicbir fold'da train ve val aylari kesismemeli ve
    hold-out aylari hicbir fold'un train'inde gorunmemeli."""
    lo, hi = holdout_months()
    for fold in build_folds():
        tr_lo, tr_hi = fold.train_months
        va_lo, va_hi = fold.val_months
        if tr_hi >= va_lo:
            raise AssertionError(f"{fold.describe()}: train val ile kesisiyor")
        if tr_hi >= lo:
            raise AssertionError(f"{fold.describe()}: train hold-out'a ({lo}-{hi}) tasiyor")
        if va_hi >= lo:
            raise AssertionError(f"{fold.describe()}: val hold-out'a ({lo}-{hi}) tasiyor")


def stability(scores: list[float]) -> dict[str, float]:
    """Fold'lar arasi kararlilik. Aylik volatilite 2.69x oynadigi icin
    ortalama kadar onemli bir model secim kriteridir."""
    arr = np.asarray(scores, dtype=float)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "worst_fold": int(np.argmin(arr)) + 1,
    }

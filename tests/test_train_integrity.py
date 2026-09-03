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


def test_load_dataset_float32_halves_memory(tmp_path, monkeypatch):
    """float32 loading must halve memory without changing the values.

    This is not cosmetic: the float64 path exhausted RAM on a 16 GB machine while
    another job held ~2 GB, which is exactly how it failed during development.
    """
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    from src.config import load_config
    from src.models import train as train_mod

    n, k = 5000, 40
    rng = np.random.default_rng(3)
    data = {"sample_id": np.arange(n, dtype=np.int32),
            "month": np.repeat(np.arange(10), n // 10).astype(np.int16),
            "target": rng.normal(0, 0.0026, n)}
    for i in range(k):
        data[f"mkt_f{i}"] = rng.normal(0, 1, n)
    tbl = pa.table(data)

    features_dir = tmp_path / "features"
    features_dir.mkdir()
    pq.write_table(tbl, features_dir / "dataset_train.parquet")

    cfg = load_config()
    monkeypatch.setitem(cfg["paths"], "features", str(features_dir))
    train_mod.load_config.cache_clear()
    monkeypatch.setattr(train_mod, "load_config", lambda: cfg)

    wide = train_mod.load_dataset("train", float32=False)
    slim = train_mod.load_dataset("train", float32=True)

    assert slim["mkt_f0"].dtype == np.float32
    assert slim["sample_id"].dtype == np.int32          # ids stay integral
    assert slim.memory_usage(deep=True).sum() < 0.6 * wide.memory_usage(deep=True).sum()
    pd.testing.assert_series_equal(
        wide["mkt_f0"].astype(np.float32), slim["mkt_f0"], check_names=False
    )


def test_load_dataset_sorts_a_shuffled_artefact(tmp_path, monkeypatch):
    """A shuffled artefact must be repaired on load, not silently accepted.

    BigQuery's list_rows() returns rows in arbitrary order. Nothing in the pipeline
    depends on row order - folds come from the month column - but a sequential read of
    a shuffled table is quietly wrong: the target autocorrelation reads +0.005 shuffled
    versus its true +0.001. This guard makes the artefact's order meaningful.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    from src.config import load_config
    from src.models import train as train_mod

    n = 2000
    rng = np.random.default_rng(11)
    order = rng.permutation(n)
    tbl = pa.table({
        "sample_id": order.astype(np.int32),
        "month": (order // 200).astype(np.int16),
        "target": rng.normal(0, 0.0026, n),
        "mkt_f0": rng.normal(0, 1, n),
    })

    features_dir = tmp_path / "features"
    features_dir.mkdir()
    pq.write_table(tbl, features_dir / "dataset_train.parquet")

    cfg = load_config()
    monkeypatch.setitem(cfg["paths"], "features", str(features_dir))
    monkeypatch.setattr(train_mod, "load_config", lambda: cfg)

    df = train_mod.load_dataset("train")
    assert df["sample_id"].is_monotonic_increasing
    assert len(df) == n

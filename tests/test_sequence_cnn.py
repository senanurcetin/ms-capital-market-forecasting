"""The tensor builder behind the sequence model. PyTorch is not needed for any of this."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from src.models import sequence_cnn as sq


def _snapshots(sample_id: int, n: int, *, mid0: float = 1.0, step: float = 1e-4) -> pd.DataFrame:
    """n snapshots, oldest first, mid rising by `step` each tick, spread 2e-4."""
    mid = mid0 + step * np.arange(n)
    return pd.DataFrame({
        "sample_id": sample_id,
        "seconds_before_predict": np.linspace(600, 0, n),
        "transaction_avgprice": mid, "transaction_volume": 2.0, "transaction_count": 1.0,
        "ask_price_1": mid + 1e-4, "bid_price_1": mid - 1e-4,
        "ask_volume_1": 3.0, "bid_volume_1": 1.0,
        "ask_price_2": mid + 2e-4, "bid_price_2": mid - 2e-4,
        "ask_volume_2": 5.0, "bid_volume_2": 5.0,
    })


def _frame(*parts: pd.DataFrame) -> pd.DataFrame:
    return pd.concat(parts, ignore_index=True)


def test_shape_ids_and_newest_snapshot_is_last():
    ids, X = sq.build_tensor(_frame(_snapshots(5, 10), _snapshots(9, 4)), window=8)
    assert X.shape == (2, 8, sq.N_CHANNELS)
    np.testing.assert_array_equal(ids, [5, 9])
    age = X[:, :, sq.CHANNELS.index("age")]
    assert age[0, -1] == pytest.approx(0.0)  # seconds_before_predict == 0 is the last step


def test_a_long_sample_keeps_its_newest_snapshots_not_its_oldest():
    df = _snapshots(1, 20)
    _, X = sq.build_tensor(df, window=6)
    age = X[0, :, sq.CHANNELS.index("age")]
    expected = (df["seconds_before_predict"].to_numpy()[-6:] / 600.0)
    np.testing.assert_allclose(age, expected, atol=1e-6)
    assert X[0, :, sq.CHANNELS.index("valid")].tolist() == [1.0] * 6


def test_a_short_sample_is_padded_with_its_oldest_snapshot_and_flagged():
    df = _snapshots(1, 3)
    _, X = sq.build_tensor(df, window=6)
    valid = X[0, :, sq.CHANNELS.index("valid")]
    assert valid.tolist() == [0, 0, 0, 1, 1, 1]
    # the padding repeats the oldest real row, so no zeros leak in as fake prices
    real = [i for i in range(sq.N_CHANNELS) if sq.CHANNELS[i] != "valid"]
    for pad in (0, 1, 2):
        np.testing.assert_array_equal(X[0, pad, real], X[0, 3, real])
    assert np.isfinite(X).all()


def test_mid_return_is_measured_against_the_newest_mid_and_ends_at_zero():
    df = _snapshots(1, 5, mid0=1.0, step=1e-3)
    _, X = sq.build_tensor(df, window=5)
    r = X[0, :, 0]
    assert r[-1] == pytest.approx(0.0)
    mid = 1.0 + 1e-3 * np.arange(5)
    np.testing.assert_allclose(r, (mid / mid[-1] - 1.0) * 1e4, atol=1e-3)


def test_the_level_of_the_price_does_not_leak_into_the_channels():
    a = sq.build_tensor(_snapshots(1, 6, mid0=1.00, step=0.0), window=6)[1]
    b = sq.build_tensor(_snapshots(1, 6, mid0=1.03, step=0.0), window=6)[1]
    # spread in bps shifts by a few parts in a thousand with the level; returns are identical
    np.testing.assert_allclose(a[:, :, 0], b[:, :, 0], atol=1e-6)
    np.testing.assert_allclose(a[:, :, 1], b[:, :, 1], rtol=0.05)


def test_the_empty_level_sentinel_is_carried_forward_not_averaged_in():
    df = _snapshots(1, 6)
    df.loc[3, ["ask_price_1", "ask_volume_1"]] = 0.0  # empty ask: price 0 AND volume 0
    _, X = sq.build_tensor(df, window=6)
    ret = X[0, :, 0]
    assert np.isfinite(ret).all()
    assert abs(ret[3] - ret[2]) < 5.0, "a zero price must not read as a 5000 bps collapse"
    assert X[0, 3, sq.CHANNELS.index("spread_bps")] == 0.0


def test_a_sample_with_an_empty_book_throughout_stays_finite():
    df = _snapshots(1, 4)
    df[["ask_price_1", "bid_price_1", "ask_volume_1", "bid_volume_1"]] = 0.0
    _, X = sq.build_tensor(df, window=4)
    assert np.isfinite(X).all()
    assert np.all(X[0, :, 0] == 0.0)


def test_samples_do_not_bleed_into_each_other():
    long_ = _snapshots(1, 6, mid0=1.0)
    other = _snapshots(2, 6, mid0=1.2)
    _, both = sq.build_tensor(_frame(long_, other), window=6)
    _, alone = sq.build_tensor(other, window=6)
    np.testing.assert_array_equal(both[1], alone[0])


def test_the_builder_refuses_a_frame_that_is_not_grouped_by_sample():
    df = _frame(_snapshots(2, 3), _snapshots(1, 3))
    with pytest.raises(ValueError, match="sorted by sample_id"):
        sq.build_tensor(df)


def test_scaler_is_fitted_on_the_given_samples_and_leaves_valid_alone():
    rng = np.random.default_rng(0)
    X = rng.normal(5.0, 3.0, size=(50, 10, sq.N_CHANNELS)).astype(np.float32)
    X[:, :, -1] = 1.0
    mu, sd = sq.fit_scaler(X)
    Z = sq.apply_scaler(X, mu, sd)
    assert abs(float(Z[:, :, 0].mean())) < 0.05 and abs(float(Z[:, :, 0].std()) - 1.0) < 0.05
    assert np.all(Z[:, :, -1] == 1.0)
    assert Z.max() <= 6.0 and Z.min() >= -6.0


def test_chunked_build_is_identical_to_the_one_piece_build():
    import pyarrow as pa

    df = _frame(*[_snapshots(i, n, mid0=1.0 + 0.01 * i)
                  for i, n in zip(range(1, 9), (3, 9, 5, 12, 1, 7, 8, 2), strict=True)])
    ids1, X1 = sq.build_tensor(df, window=6)
    for rows in (1, 4, 10, 1000):
        ids2, X2 = sq.build_tensor_chunked(pa.Table.from_pandas(df), window=6, rows_per_chunk=rows)
        np.testing.assert_array_equal(ids1, ids2)
        np.testing.assert_array_equal(X1, X2)

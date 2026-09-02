"""Backtest testleri - ozellikle look-ahead ve maliyet davranisi."""
import numpy as np
import pytest

from src.evaluation.backtesting import (
    backtest, cost_sensitivity, make_signals, sweep_trade_fraction, threshold_from_quantile,
)

rng = np.random.default_rng(11)


def test_signals_three_states():
    p = np.array([-0.5, -0.05, 0.0, 0.05, 0.5])
    s = make_signals(p, 0.1)
    assert list(s) == [-1, 0, 0, 0, 1]


def test_threshold_from_quantile_trades_expected_fraction():
    p = rng.normal(0, 1, 10_000)
    t = threshold_from_quantile(p, 0.2)
    assert make_signals(p, t).astype(bool).mean() == pytest.approx(0.2, abs=0.01)


def test_perfect_prediction_is_profitable():
    a = rng.normal(0, 0.0026, 5000)
    r = backtest(a, a, trade_fraction=0.2, cost_bps=0.0)
    assert r["total_return"] > 0 and r["win_rate"] == pytest.approx(1.0)


def test_inverted_prediction_loses():
    a = rng.normal(0, 0.0026, 5000)
    assert backtest(-a, a, trade_fraction=0.2, cost_bps=0.0)["total_return"] < 0


def test_pure_noise_is_within_sampling_error():
    """Sinyalsiz tahmin kar uretmemeli.

    Sabit bir esik kullanilamaz: n bagimsiz bahsin toplami sqrt(n) ile buyur,
    yani 4000 islemde dogal std = sqrt(4000) * 0.0026 ~ 0.164. Dogru test,
    toplamin bu orneklem hatasinin 3 katini asmamasidir.
    """
    a = rng.normal(0, 0.0026, 20_000)
    r = backtest(rng.normal(0, 1, 20_000), a, trade_fraction=0.2, cost_bps=0.0)
    sampling_sd = np.sqrt(r["n_trades"]) * a.std(ddof=1)
    assert abs(r["total_return"]) < 3 * sampling_sd


def test_costs_monotonically_reduce_return():
    a = rng.normal(0, 0.0026, 5000)
    p = a + rng.normal(0, 0.002, 5000)
    df = cost_sensitivity(p, a, costs_bps=(0.0, 1.0, 2.0, 5.0))
    assert df["total_return"].is_monotonic_decreasing


def test_zero_trades_returns_zeroed_metrics():
    a = rng.normal(0, 0.0026, 100)
    r = backtest(np.zeros(100), a, threshold=1.0)
    assert r["n_trades"] == 0 and r["sharpe"] == 0.0 and r["total_return"] == 0.0


def test_drawdown_is_non_positive():
    a = rng.normal(0, 0.0026, 3000)
    r = backtest(a + rng.normal(0, 0.002, 3000), a, trade_fraction=0.3)
    assert r["max_drawdown"] <= 0


def test_sweep_returns_row_per_fraction():
    a = rng.normal(0, 0.0026, 2000)
    df = sweep_trade_fraction(a + rng.normal(0, 0.002, 2000), a)
    assert len(df) == 5 and df["turnover"].is_monotonic_increasing


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        backtest(np.zeros(10), np.zeros(11))

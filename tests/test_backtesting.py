"""Backtest tests - especially the no-look-ahead and cost behaviour."""
import numpy as np
import pytest
from src.evaluation.backtesting import (
    backtest,
    cost_sensitivity,
    make_signals,
    sweep_trade_fraction,
    threshold_from_quantile,
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
    """A signal-free prediction must not produce profit.

    A fixed threshold will not do: the sum of n independent bets grows with
    sqrt(n), so across 4000 trades the natural standard deviation is
    sqrt(4000) * 0.0026 ~ 0.164. The correct test is that the total stays within
    three times that sampling error.
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


def test_quantile_backtest_trades_tied_magnitudes_with_stable_index_order():
    """Ties must not make a 50% trade budget silently select zero trades."""
    pred = np.array([1.0, -1.0, 1.0, -1.0])
    actual = np.array([1.0, 2.0, 3.0, 4.0])

    half = backtest(pred, actual, trade_fraction=0.5)
    full = backtest(pred, actual, trade_fraction=1.0)

    assert half["threshold"] == 1.0
    assert half["n_trades"] == 2
    assert half["turnover"] == pytest.approx(0.5)
    # Stable tie-breaking picks positions 0 (long) and 1 (short).
    assert half["total_return"] == pytest.approx(-1.0)
    assert full["n_trades"] == 4
    assert full["turnover"] == 1.0
    assert full["total_return"] == pytest.approx(-2.0)


def test_quantile_backtest_does_not_trade_zero_predictions_to_fill_budget():
    pred = np.array([0.0, 0.0, 2.0, -2.0])
    actual = np.array([3.0, -3.0, 0.01, -0.02])

    result = backtest(pred, actual, trade_fraction=1.0)
    assert result["n_trades"] == 2
    assert result["turnover"] == 0.5
    assert result["total_return"] == pytest.approx(0.03)
    assert backtest(np.zeros(4), actual, trade_fraction=1.0)["n_trades"] == 0


def test_quantile_ties_do_not_look_ahead_at_realised_returns():
    pred = np.array([1.0, -1.0, 1.0, -1.0])
    first = backtest(pred, np.array([1.0, 2.0, 100.0, 100.0]), trade_fraction=0.5)
    second = backtest(pred, np.array([1.0, 2.0, -100.0, -100.0]), trade_fraction=0.5)

    assert first["n_trades"] == second["n_trades"] == 2
    assert first["total_return"] == second["total_return"] == pytest.approx(-1.0)


def test_tied_trades_pay_exact_per_trade_cost():
    pred = np.array([-1.0, -1.0, 1.0, 1.0])
    actual = np.array([0.01, -0.02, 0.03, -0.04])

    free = backtest(pred, actual, trade_fraction=0.5, cost_bps=0)
    charged = backtest(pred, actual, trade_fraction=0.5, cost_bps=5)

    assert charged["n_trades"] == free["n_trades"] == 2
    assert free["total_return"] - charged["total_return"] == pytest.approx(2 * 5 / 10_000)


def test_explicit_threshold_keeps_strict_inequality_at_ties():
    pred = np.array([1.0, -1.0, 1.0, -1.0])
    actual = np.array([1.0, 1.0, 1.0, 1.0])
    assert backtest(pred, actual, threshold=1.0)["n_trades"] == 0
    assert backtest(pred, actual, threshold=0.99)["n_trades"] == 4

def test_backtest_rejects_nonfinite_predictions_and_returns():
    for value in (np.nan, np.inf, -np.inf):
        with pytest.raises(ValueError, match="pred must contain only finite"):
            backtest(np.array([1.0, value]), np.array([0.1, 0.2]))
        with pytest.raises(ValueError, match="actual must contain only finite"):
            backtest(np.array([1.0, 0.0]), np.array([0.1, value]))


def test_backtest_rejects_nonfinite_costs_thresholds_and_empty_inputs():
    with pytest.raises(ValueError, match="nonempty 1-D"):
        backtest(np.array([]), np.array([]))
    for bad in (-1.0, np.nan, np.inf):
        with pytest.raises(ValueError, match="cost_bps"):
            backtest(np.array([0.0]), np.array([0.0]), cost_bps=bad)
        with pytest.raises(ValueError, match="threshold"):
            backtest(np.array([0.0]), np.array([0.0]), threshold=bad)
    with pytest.raises(ValueError, match="pred must contain only finite"):
        threshold_from_quantile(np.array([1.0, np.nan]), 0.5)

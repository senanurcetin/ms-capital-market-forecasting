"""Research-only backtesting module.

THIS IS NOT AN INVESTMENT TOOL. Its purpose is to assess whether the model's
predictions carry statistically meaningful ranking power, framed in trading terms.

A MODELLING CHOICE FORCED BY THE DATA:
    Target autocorrelation between consecutive samples is ~0 (lag1: 0.0011,
    lag5: 0.0021) and there is no symbol column, so positions are NOT carried:
    each sample is modelled as an independent bet. "Turnover" is therefore
    defined as the fraction of samples that are traded.

NO LOOK-AHEAD: every prediction uses only its own window's history, and the
backtest is run exclusively on the HOLD-OUT months.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_signals(pred: np.ndarray, threshold: float) -> np.ndarray:
    """pred > +t -> +1 (BUY), pred < -t -> -1 (SELL), otherwise 0 (HOLD)."""
    p = np.asarray(pred, dtype=np.float64)
    return np.where(p > threshold, 1, np.where(p < -threshold, -1, 0)).astype(np.int8)


def threshold_from_quantile(pred: np.ndarray, trade_fraction: float) -> float:
    """Threshold that trades the most extreme `trade_fraction` of predictions.

    Used instead of a fixed absolute threshold because cosine similarity is
    scale-invariant, so prediction MAGNITUDE is not calibrated - an absolute cut
    would be meaningless.
    """
    if not 0 < trade_fraction <= 1:
        raise ValueError("trade_fraction must be in (0, 1]")
    return float(np.quantile(np.abs(pred), 1.0 - trade_fraction))


def backtest(
    pred: np.ndarray,
    actual: np.ndarray,
    *,
    threshold: float | None = None,
    trade_fraction: float = 0.2,
    cost_bps: float = 0.0,
) -> dict:
    """Per-signal return = direction * realised return - transaction cost."""
    p = np.asarray(pred, dtype=np.float64)
    a = np.asarray(actual, dtype=np.float64)
    if p.shape != a.shape:
        raise ValueError(f"shape mismatch: {p.shape} vs {a.shape}")

    t = threshold_from_quantile(p, trade_fraction) if threshold is None else threshold
    sig = make_signals(p, t)
    traded = sig != 0
    n_trades = int(traded.sum())

    cost = cost_bps / 10_000.0
    pnl = sig * a - np.abs(sig) * cost

    if n_trades == 0:
        return {"threshold": t, "n_trades": 0, "turnover": 0.0, "total_return": 0.0,
                "mean_return": 0.0, "volatility": 0.0, "sharpe": 0.0,
                "max_drawdown": 0.0, "win_rate": 0.0, "cost_bps": cost_bps}

    equity = np.cumsum(pnl)
    peak = np.maximum.accumulate(equity)
    drawdown = equity - peak
    trade_pnl = pnl[traded]
    vol = float(trade_pnl.std(ddof=1)) if n_trades > 1 else 0.0

    return {
        "threshold": float(t),
        "n_trades": n_trades,
        "turnover": float(traded.mean()),
        "total_return": float(equity[-1]),
        "mean_return": float(trade_pnl.mean()),
        "volatility": vol,
        # Per-trade Sharpe (not annualised): samples are independent bets.
        "sharpe": float(trade_pnl.mean() / vol) if vol > 0 else 0.0,
        "max_drawdown": float(drawdown.min()),
        "win_rate": float((trade_pnl > 0).mean()),
        "cost_bps": cost_bps,
        "equity_curve": equity,
    }


def cost_sensitivity(
    pred: np.ndarray, actual: np.ndarray, *,
    trade_fraction: float = 0.2, costs_bps: tuple[float, ...] = (0.0, 1.0, 2.0, 5.0),
) -> pd.DataFrame:
    """How the strategy holds up as costs rise - a direct test of real signal strength."""
    rows = []
    for c in costs_bps:
        r = backtest(pred, actual, trade_fraction=trade_fraction, cost_bps=c)
        r.pop("equity_curve", None)
        rows.append(r)
    return pd.DataFrame(rows)


def sweep_trade_fraction(
    pred: np.ndarray, actual: np.ndarray, *,
    fractions: tuple[float, ...] = (0.05, 0.1, 0.2, 0.5, 1.0), cost_bps: float = 1.0,
) -> pd.DataFrame:
    """Does signal strength increase with prediction rank? The most extreme
    predictions should be the most profitable; if not, the model has no ranking power."""
    rows = []
    for f in fractions:
        r = backtest(pred, actual, trade_fraction=f, cost_bps=cost_bps)
        r.pop("equity_curve", None)
        rows.append(r)
    return pd.DataFrame(rows)

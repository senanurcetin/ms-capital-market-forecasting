"""Page 6 - Backtest (research only)."""
import pandas as pd
import streamlit as st

from streamlit_app.lib import FEATURES_DIR, missing, page_header

st.set_page_config(page_title="Backtesting", layout="wide")
page_header("Backtest", "Assessing predictions in a trading-like framing")
st.warning(
    "This is NOT a strategy recommendation. The goal is to measure the model's ranking "
    "power and its robustness to transaction costs."
)

cost_path = FEATURES_DIR / "backtest_cost_sensitivity.csv"
sweep_path = FEATURES_DIR / "backtest_trade_fraction.csv"
equity_path = FEATURES_DIR / "backtest_equity.csv"

if not cost_path.exists():
    missing("Backtest results", "python -m src.models.finalize")
    st.stop()

cost = pd.read_csv(cost_path)
st.subheader("Transaction-cost sensitivity")
st.dataframe(cost, use_container_width=True)
if {"cost_bps", "total_return"} <= set(cost.columns):
    st.line_chart(cost.set_index("cost_bps")["total_return"])
st.caption(
    "If the signal is real, returns should decay smoothly as costs rise; collapsing at "
    "even a small cost means the ranking power is weak."
)

if sweep_path.exists():
    sweep = pd.read_csv(sweep_path)
    st.subheader("Performance by traded fraction")
    st.dataframe(sweep, use_container_width=True)
    st.caption(
        "The threshold is a tail percentile of the prediction distribution, not an "
        "absolute cut - cosine is scale-invariant, so magnitudes are not calibrated."
    )

if equity_path.exists():
    eq = pd.read_csv(equity_path)
    st.subheader("Cumulative return (hold-out months 65-70)")
    st.line_chart(eq.iloc[:, 0])
    st.caption("The backtest runs ONLY on the hold-out months - no look-ahead.")

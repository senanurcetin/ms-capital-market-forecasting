"""Page 6 - Backtest (research only)."""
import sys
from pathlib import Path

# Put the repository root on sys.path before importing anything from it.
#
# `python -m streamlit` silently adds the working directory; a bare `streamlit run` - which
# is what Streamlit Community Cloud executes - adds the MAIN SCRIPT'S directory instead. So
# `streamlit_app/` lands on the path and the repository root does not, and every
# `from streamlit_app.lib import ...` below fails with ModuleNotFoundError. It works locally
# and breaks on deploy, which is the worst place to find out.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

from streamlit_app.lib import load_csv, missing, page_header

st.set_page_config(page_title="MSCapital · Backtesting", page_icon="📈", layout="wide")
page_header("Backtest", "Assessing predictions in a trading-like framing")
st.warning(
    "This is NOT a strategy recommendation. The goal is to measure the model's ranking "
    "power and its robustness to transaction costs."
)

cost = load_csv("backtest_cost_sensitivity.csv")
if cost is None:
    missing("Backtest results", "python -m src.models.finalize")
    st.stop()

st.subheader("Transaction-cost sensitivity")
st.dataframe(cost, width="stretch")
if {"cost_bps", "total_return"} <= set(cost.columns):
    st.line_chart(cost.set_index("cost_bps")["total_return"])
st.caption(
    "If the signal is real, returns should decay smoothly as costs rise; collapsing at "
    "even a small cost means the ranking power is weak."
)

sweep = load_csv("backtest_trade_fraction.csv")
if sweep is not None:
    st.subheader("Performance by traded fraction")
    st.dataframe(sweep, width="stretch")
    st.caption(
        "The threshold is a tail percentile of the prediction distribution, not an "
        "absolute cut - cosine is scale-invariant, so magnitudes are not calibrated."
    )

eq = load_csv("backtest_equity.csv")
if eq is not None:
    st.subheader("Cumulative return (hold-out months 65-70)")
    st.line_chart(eq.iloc[:, 0])
    st.caption("The backtest runs ONLY on the hold-out months - no look-ahead.")

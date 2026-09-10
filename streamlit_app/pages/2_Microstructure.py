"""Page 2 - Market microstructure."""
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

from streamlit_app.lib import histogram, load_features, missing, page_header

st.set_page_config(page_title="MSCapital · Microstructure", page_icon="📈", layout="wide")
page_header("Market Microstructure", "Order book, order flow and trade dynamics")

COLS = [
    "month", "target",
    "mkt_rel_spread_last", "mkt_spread_mean_clean", "mkt_depth_imb1_last",
    "mkt_depth_imb12_last", "mkt_total_depth_last", "mkt_mid_std_60s", "mkt_mid_std_600s",
    "mkt_empty_bid_share", "mkt_empty_ask_share",
    "ord_ofi_60s", "ord_cancel_new_ratio_60s", "ord_new_volume_imbalance_60s",
    "txn_volume_imbalance_60s", "txn_intensity_60s", "txn_volume_rate_60s",
]
df = load_features(n_rows=40_000, columns=COLS)
if df is None:
    missing("Feature set", "python -m src.features.assemble")
    st.stop()

tab1, tab2, tab3 = st.tabs(["Order book", "Order flow", "Trades"])

with tab1:
    c1, c2 = st.columns(2)
    c1.subheader("Relative spread (bps)")
    c1.bar_chart(histogram((df["mkt_rel_spread_last"] * 1e4).clip(0, 60), label="bps"))
    c2.subheader("Depth imbalance (L1)")
    c2.bar_chart(histogram(df["mkt_depth_imb1_last"], label="imbalance"))
    st.subheader("Volatility: 60 s vs 600 s")
    st.line_chart(df.groupby("month")[["mkt_mid_std_60s", "mkt_mid_std_600s"]].mean())
    st.caption("The market window is 600 s; order and transaction are 60 s - measured, not assumed.")
    m1, m2 = st.columns(2)
    m1.metric("Empty bid side", f"{df['mkt_empty_bid_share'].mean() * 100:.2f}%")
    m2.metric("Empty ask side", f"{df['mkt_empty_ask_share'].mean() * 100:.2f}%")
    st.caption(
        "price = 0 is not a price but a 'this level is empty' sentinel (it always "
        "comes with volume = 0). Left uncleaned, it corrupts spread and mid."
    )

with tab2:
    c1, c2 = st.columns(2)
    c1.subheader("Order flow imbalance (60s)")
    c1.bar_chart(histogram(df["ord_ofi_60s"], label="OFI"))
    c2.subheader("Cancel-to-new order ratio")
    c2.bar_chart(histogram(df["ord_cancel_new_ratio_60s"].clip(0, 3), label="cancel / new"))
    st.caption("side 0 = BID, 1 = ASK; order_action 0 = NEW, 1 = CANCEL (resolved empirically).")

with tab3:
    c1, c2 = st.columns(2)
    c1.subheader("Trade volume imbalance")
    c1.bar_chart(histogram(df["txn_volume_imbalance_60s"], label="signed volume"))
    c2.subheader("Trade intensity (trades/s)")
    c2.bar_chart(histogram(df["txn_intensity_60s"].clip(0, 10), label="trades / s"))

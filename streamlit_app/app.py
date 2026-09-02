"""MSCapital - Financial Market Intelligence Dashboard (Page 1: Overview)."""
import streamlit as st

from streamlit_app.lib import (
    api_get,
    feature_columns,
    load_features,
    load_results_table,
    missing,
    page_header,
)

st.set_page_config(page_title="MSCapital | Overview", layout="wide")
page_header(
    "MSCapital - Market Intelligence",
    "Short-horizon return prediction from 60 s of order/trade flow and 600 s of book history",
)

api = api_get("/health")
c1, c2, c3 = st.columns(3)
c1.metric("API", "ok" if api and api.get("status") == "ok" else "degraded / offline")
cols = feature_columns()
c2.metric("Features", len(cols) - 3 if cols else 0)
res = load_results_table()
c3.metric("Best cosine", f"{res['cosine_mean'].max():+.4f}" if res is not None else "-")

st.divider()

df = load_features(
    n_rows=20_000,
    columns=[
        "sample_id",
        "month",
        "target",
        "mkt_mid_last",
        "mkt_rel_spread_last",
        "mkt_depth_imb1_last",
        "mkt_mid_std_60s",
        "ord_ofi_60s",
        "txn_volume_imbalance_60s",
        "txn_intensity_60s",
    ],
)
if df is None:
    missing("Feature set", "python -m src.features.assemble")
    st.stop()

st.subheader("Market state (first 20,000 samples)")
k = st.columns(6)
k[0].metric("Spread (rel)", f"{df['mkt_rel_spread_last'].mean() * 1e4:.1f} bps")
k[1].metric("Depth imbalance", f"{df['mkt_depth_imb1_last'].mean():+.3f}")
k[2].metric("Volatility (60s)", f"{df['mkt_mid_std_60s'].mean() * 1e4:.1f} bps")
k[3].metric("Order flow imbalance", f"{df['ord_ofi_60s'].mean():+.3f}")
k[4].metric("Trade intensity", f"{df['txn_intensity_60s'].mean():.2f}/s")
k[5].metric("Target std", f"{df['target'].std() * 1e4:.1f} bps")

st.divider()
left, right = st.columns(2)
with left:
    st.subheader("Target distribution")
    st.bar_chart(df["target"].clip(-0.01, 0.01).value_counts(bins=60).sort_index())
    st.caption("Median is exactly 0 (5.5% exact zeros) - a tick-size artefact.")
with right:
    st.subheader("Target volatility by month")
    st.line_chart(df.groupby("month")["target"].std())
    st.caption("Across full train the monthly std swings by 2.69x - regime shift.")

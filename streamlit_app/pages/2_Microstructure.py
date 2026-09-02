"""Sayfa 2 - Market microstructure."""
import streamlit as st

from streamlit_app.lib import load_features, missing, page_header

st.set_page_config(page_title="Microstructure", layout="wide")
page_header("Market Microstructure", "Defter, emir akisi ve islem dinamikleri")

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
    missing("Feature seti", "python -m src.features.assemble")
    st.stop()

tab1, tab2, tab3 = st.tabs(["Defter", "Emir akisi", "Islemler"])

with tab1:
    c1, c2 = st.columns(2)
    c1.subheader("Goreli spread (bps)")
    c1.bar_chart((df["mkt_rel_spread_last"] * 1e4).clip(0, 60).value_counts(bins=50).sort_index())
    c2.subheader("Derinlik dengesizligi (L1)")
    c2.bar_chart(df["mkt_depth_imb1_last"].value_counts(bins=50).sort_index())
    st.subheader("Volatilite: 60 sn vs 600 sn")
    st.line_chart(df.groupby("month")[["mkt_mid_std_60s", "mkt_mid_std_600s"]].mean())
    st.caption("Market penceresi 600 sn; order ve transaction 60 sn - farkli olculdu.")
    m1, m2 = st.columns(2)
    m1.metric("Bos bid tarafi", f"{df['mkt_empty_bid_share'].mean() * 100:.2f}%")
    m2.metric("Bos ask tarafi", f"{df['mkt_empty_ask_share'].mean() * 100:.2f}%")
    st.caption(
        "price = 0 bir fiyat degil, 'bu seviye bos' sentinel'idir (her zaman volume = 0 "
        "ile gelir). Temizlenmezse spread ve mid coplenir."
    )

with tab2:
    c1, c2 = st.columns(2)
    c1.subheader("Order flow imbalance (60s)")
    c1.bar_chart(df["ord_ofi_60s"].value_counts(bins=50).sort_index())
    c2.subheader("Iptal / yeni emir orani")
    c2.bar_chart(df["ord_cancel_new_ratio_60s"].clip(0, 3).value_counts(bins=50).sort_index())
    st.caption("side 0 = BID, 1 = ASK; order_action 0 = NEW, 1 = CANCEL (ampirik cozuldu).")

with tab3:
    c1, c2 = st.columns(2)
    c1.subheader("Islem hacim dengesizligi")
    c1.bar_chart(df["txn_volume_imbalance_60s"].value_counts(bins=50).sort_index())
    c2.subheader("Islem yogunlugu (islem/sn)")
    c2.bar_chart(df["txn_intensity_60s"].clip(0, 10).value_counts(bins=50).sort_index())

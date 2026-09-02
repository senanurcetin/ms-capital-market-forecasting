"""Sayfa 6 - Backtest (arastirma amacli)."""
import pandas as pd
import streamlit as st

from streamlit_app.lib import FEATURES_DIR, missing, page_header

st.set_page_config(page_title="Backtesting", layout="wide")
page_header("Backtest", "Tahminlerin islem-benzeri bir cerceveede degerlendirilmesi")
st.warning(
    "Bu bir strateji onerisi DEGILDIR. Amac, modelin siralama gucunu ve islem "
    "maliyetine dayanikliligini olcmektir."
)

cost_path = FEATURES_DIR / "backtest_cost_sensitivity.csv"
sweep_path = FEATURES_DIR / "backtest_trade_fraction.csv"
equity_path = FEATURES_DIR / "backtest_equity.csv"

if not cost_path.exists():
    missing("Backtest sonuclari", "python -m src.evaluation.run_backtest")
    st.stop()

cost = pd.read_csv(cost_path)
st.subheader("Islem maliyetine duyarlilik")
st.dataframe(cost, use_container_width=True)
if {"cost_bps", "total_return"} <= set(cost.columns):
    st.line_chart(cost.set_index("cost_bps")["total_return"])
st.caption(
    "Sinyal gercekse getiri maliyet arttikca duzgun azalmali; kucuk maliyette bile "
    "cokuyorsa siralama gucu zayif demektir."
)

if sweep_path.exists():
    sweep = pd.read_csv(sweep_path)
    st.subheader("Islem yapilan orana gore performans")
    st.dataframe(sweep, use_container_width=True)
    st.caption(
        "Esik mutlak degil, tahmin dagiliminin kuyruk yuzdesi olarak secilir - cosine "
        "olcek-degismez oldugu icin tahmin buyuklugu kalibre degildir."
    )

if equity_path.exists():
    eq = pd.read_csv(equity_path)
    st.subheader("Kumulatif getiri (hold-out ay 65-70)")
    st.line_chart(eq.iloc[:, 0])
    st.caption("Backtest YALNIZ hold-out aylarinda calistirilir - look-ahead yok.")

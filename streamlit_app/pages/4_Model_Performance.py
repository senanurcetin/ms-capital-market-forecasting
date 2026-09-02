"""Sayfa 4 - Model karsilastirmasi ve temporal stabilite."""
import pandas as pd
import streamlit as st

from streamlit_app.lib import load_results_table, load_summary, missing, page_header

st.set_page_config(page_title="Model Performance", layout="wide")
page_header("Model Performansi", "Walk-forward validation - ana metrik cosine similarity")

table = load_results_table()
summary = load_summary()
if table is None or summary is None:
    missing("Walk-forward sonuclari", "python -m src.models.train")
    st.stop()

st.subheader("Model karsilastirmasi")
fmt = {c: "{:+.5f}" for c in table.columns if c.startswith("cosine")}
st.dataframe(table.style.format(fmt), use_container_width=True)
st.caption(
    "Cosine OLCEK-degismez ama KAYDIRMA-degismez degildir: sabit tahmin eden 'mean' "
    "modeli negatif skor uretir. Model secimi ortalamaya oldugu kadar fold'lar arasi "
    "std'ye de bakilarak yapilir."
)

rows = []
for model, blk in summary.items():
    for r in blk.get("per_fold", []):
        value = r.get("cosine", r.get("ensemble_score"))
        if value is not None:
            rows.append({"model": model, "fold": r.get("fold"), "cosine": value})

if rows:
    per_fold = pd.DataFrame(rows).pivot(index="fold", columns="model", values="cosine")
    st.subheader("Fold bazinda temporal stabilite")
    st.line_chart(per_fold)
    st.caption(
        "Aylik hedef volatilitesi 2.69x oynadigi icin fold'lar arasi kararlilik, "
        "ortalama kadar onemli bir model secim kriteri."
    )

if "ensemble" in summary:
    ens = summary["ensemble"]
    st.metric(
        "Ensemble en iyi tek modeli gectigi fold",
        f"{ens.get('beats_best_single_in_folds', 0)} / {ens.get('n_folds', 0)}",
    )
    st.caption(
        "Agirliklar grid search ile degil kapali formda bulunur: cosine olcek-degismez "
        "oldugundan y'nin model tahminlerinin span'ine dik izdusumu optimaldir, o da "
        "OLS cozumudur."
    )

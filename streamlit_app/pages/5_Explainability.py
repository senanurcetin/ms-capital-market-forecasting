"""Sayfa 5 - Aciklanabilirlik (SHAP)."""
import pandas as pd
import streamlit as st

from streamlit_app.lib import MODELS_DIR, missing, page_header

st.set_page_config(page_title="Explainability", layout="wide")
page_header("Aciklanabilirlik", "Model bu tahmini neden verdi?")

global_path = MODELS_DIR / "current" / "shap_global.csv"
local_path = MODELS_DIR / "current" / "shap_local_examples.csv"

if not global_path.exists():
    missing("SHAP ciktilari", "python -m src.evaluation.explain")
    st.stop()

glob = pd.read_csv(global_path)
name_col, value_col = glob.columns[0], glob.columns[1]

st.subheader("Global feature onemi (ortalama |SHAP|)")
top = glob.head(25).set_index(name_col)
st.bar_chart(top[value_col])
st.caption(
    "Feature aileleri: mkt_ = defter (600 sn), ord_ = emir akisi (60 sn), "
    "txn_ = gerceklesen islemler (60 sn)."
)

with st.expander("Tam liste"):
    st.dataframe(glob, use_container_width=True)

if local_path.exists():
    st.subheader("Ornek tahmin aciklamasi (local SHAP)")
    local = pd.read_csv(local_path)
    ids = local["sample_id"].unique().tolist()
    chosen = st.selectbox("sample_id", ids)
    sel = local[local["sample_id"] == chosen].sort_values("shap_value", key=abs, ascending=False)
    st.bar_chart(sel.head(15).set_index("feature")["shap_value"])
    st.dataframe(sel.head(30), use_container_width=True)

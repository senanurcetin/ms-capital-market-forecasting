"""Page 5 - Explainability (SHAP)."""
import pandas as pd
import streamlit as st

from streamlit_app.lib import MODELS_DIR, missing, page_header

st.set_page_config(page_title="Explainability", layout="wide")
page_header("Explainability", "Why did the model make this prediction?")

global_path = MODELS_DIR / "current" / "shap_global.csv"
local_path = MODELS_DIR / "current" / "shap_local_examples.csv"

if not global_path.exists():
    missing("SHAP output", "python -m src.evaluation.explain")
    st.stop()

glob = pd.read_csv(global_path)
name_col, value_col = glob.columns[0], glob.columns[1]

st.subheader("Global feature importance (mean |SHAP|)")
top = glob.head(25).set_index(name_col)
st.bar_chart(top[value_col])
st.caption(
    "Feature families: mkt_ = order book (600 s), ord_ = order flow (60 s), "
    "txn_ = executed trades (60 s)."
)

with st.expander("Full list"):
    st.dataframe(glob, use_container_width=True)

if local_path.exists():
    st.subheader("Individual prediction explanation (local SHAP)")
    local = pd.read_csv(local_path)
    ids = local["sample_id"].unique().tolist()
    chosen = st.selectbox("sample_id", ids)
    sel = local[local["sample_id"] == chosen].sort_values("shap_value", key=abs, ascending=False)
    st.bar_chart(sel.head(15).set_index("feature")["shap_value"])
    st.dataframe(sel.head(30), use_container_width=True)

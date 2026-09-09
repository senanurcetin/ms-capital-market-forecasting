"""Page 5 - Explainability (SHAP).

Reads through lib.load_csv, which checks the local pipeline output first and the exported
bundle second. This page used to address MODELS_DIR directly, so it alone went blank away
from the pipeline - the bundle carries both SHAP files and the page could not see them.
"""
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

st.set_page_config(page_title="Explainability", layout="wide")
page_header("Explainability", "Why did the model make this prediction?")

glob = load_csv("shap_global.csv")
if glob is None:
    missing("SHAP output", "python -m src.evaluation.explain")
    st.stop()

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

local = load_csv("shap_local_examples.csv")
if local is not None:
    st.subheader("Individual prediction explanation (local SHAP)")
    ids = local["sample_id"].unique().tolist()
    chosen = st.selectbox("sample_id", ids)
    sel = local[local["sample_id"] == chosen].sort_values("shap_value", key=abs, ascending=False)
    st.bar_chart(sel.head(15).set_index("feature")["shap_value"])
    st.dataframe(sel.head(30), use_container_width=True)

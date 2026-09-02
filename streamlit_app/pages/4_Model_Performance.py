"""Page 4 - Model comparison and temporal stability."""
import pandas as pd
import streamlit as st

from streamlit_app.lib import load_results_table, load_summary, missing, page_header

st.set_page_config(page_title="Model Performance", layout="wide")
page_header("Model Performance", "Walk-forward validation - primary metric: cosine similarity")

table = load_results_table()
summary = load_summary()
if table is None or summary is None:
    missing("Walk-forward results", "python -m src.models.train")
    st.stop()

st.subheader("Model comparison")
fmt = {c: "{:+.5f}" for c in table.columns if c.startswith("cosine")}
st.dataframe(table.style.format(fmt), use_container_width=True)
st.caption(
    "Cosine is SCALE-invariant but NOT SHIFT-invariant: the constant-prediction 'mean' "
    "model scores negative. Model selection weighs across-fold std as heavily as the mean."
)

rows = []
for model, blk in summary.items():
    for r in blk.get("per_fold", []):
        value = r.get("cosine", r.get("ensemble_score"))
        if value is not None:
            rows.append({"model": model, "fold": r.get("fold"), "cosine": value})

if rows:
    per_fold = pd.DataFrame(rows).pivot(index="fold", columns="model", values="cosine")
    st.subheader("Temporal stability by fold")
    st.line_chart(per_fold)
    st.caption(
        "Because monthly target volatility swings by 2.69x, across-fold stability is "
        "as important a selection criterion as the mean."
    )

if "ensemble" in summary:
    ens = summary["ensemble"]
    st.metric(
        "Folds where the ensemble beat the best single model",
        f"{ens.get('beats_best_single_in_folds', 0)} / {ens.get('n_folds', 0)}",
    )
    st.caption(
        "The weights come from a closed form, not a grid search: because cosine is "
        "scale-invariant, the optimum is the orthogonal projection of y onto the span of "
        "the model predictions - which is exactly the OLS solution."
    )

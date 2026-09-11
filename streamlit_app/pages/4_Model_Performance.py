"""Page 4 - Model comparison and temporal stability."""
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

import pandas as pd
import streamlit as st

from streamlit_app.lib import load_results_table, load_summary, missing, page_header

st.set_page_config(page_title="MSCapital · Model performance", page_icon="📈", layout="wide")
page_header("Model Performance", "Walk-forward validation - primary metric: cosine similarity")

table = load_results_table()
summary = load_summary()
if table is None or summary is None:
    missing("Walk-forward results", "python -m src.models.train")
    st.stop()

st.subheader("Model comparison")
fmt = {c: "{:+.5f}" for c in table.columns if c.startswith("cosine")}
st.dataframe(table.style.format(fmt), width="stretch")
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
    # Counted from the per-fold record rather than read from summary keys. It used to
    # look for `beats_best_single_in_folds` and `n_folds`, neither of which the file
    # carries, so both fell through to their defaults and the page displayed "0 / 0" -
    # the strongest number on it, rendered as its own opposite.
    folds = summary["ensemble"].get("per_fold", [])
    won = sum(1 for f in folds if f.get("beats_best_single"))
    st.metric(
        "Folds where the ensemble beat the best single model",
        f"{won} / {len(folds)}",
    )
    st.caption(
        "The weights come from a closed form, not a grid search: because cosine is "
        "scale-invariant, the optimum is the orthogonal projection of y onto the span of "
        "the model predictions - which is exactly the OLS solution."
    )

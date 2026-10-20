"""Research provenance and a downloadable integrity report, computed without cloud access."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import streamlit as st
from src.evaluation.evidence import audit

from streamlit_app.lib import BUNDLED, find, page_header

st.set_page_config(page_title="MSCapital · NEXUS Evidence", page_icon=":material/fact_check:", layout="wide")
page_header("Research evidence", "Source provenance, model identity and reproducibility")

st.info(
    "Historical research snapshot. Opening this page makes no Kaggle or BigQuery calls. "
    "A file hash identifies bytes; it does not independently validate the experiment."
)

st.subheader("Which files are you viewing?")
sources = []
for name, scope in (
    ("walkforward_summary.csv", "Ensemble · five walk-forward folds"),
    ("holdout_metrics.json", "Single LightGBM · months 65–70"),
    ("leaderboard.json", "Historical external score · v4 ensemble"),
    ("model_meta.json", "Default served prediction artifact"),
    ("feature_sample.parquet", "Exploration sample · not the full dataset"),
):
    path = find(name)
    sources.append({
        "Artifact": name, "Scope": scope,
        "Source": "Bundled snapshot" if path and path.parent == BUNDLED else "Local pipeline" if path else "Unavailable",
    })
st.dataframe(pd.DataFrame(sources), hide_index=True, width="stretch")
st.caption(
    "Loaders prefer local pipeline files over the bundled snapshot, independently per file. "
    "Mixed sources can therefore describe different runs. The audit below covers only results/."
)

st.subheader("Bundled evidence integrity")
report = audit(BUNDLED)
if report["status"] == "pass":
    st.success(f"Headline metrics reconcile across JSON, CSV and {report['fold_count']} fold records.")
else:
    st.warning("Evidence checks failed. Inspect the report before quoting metrics.")
    for error in report["errors"]:
        st.write(error)

model = report["served_model"]
st.markdown(
    f"**Bundled inference:** {model['name']} {model['version']}. "
    f"**Leaderboard capture:** {report['leaderboard_captured']}. "
    "The default LightGBM v3 and submitted v4 ensemble have different evaluation scopes."
)
st.download_button(
    "Download evidence report (JSON)", json.dumps(report, indent=2, allow_nan=False),
    file_name="mscapital-evidence.json", mime="application/json",
)
with st.expander("Artifact inventory and SHA-256 hashes"):
    st.dataframe(pd.DataFrame(report["files"]), hide_index=True, width="stretch")

st.subheader("Reproduce the research")
st.markdown(
    "[Reproducibility guide](https://github.com/senanurcetin/ms-capital-market-forecasting/blob/main/docs/REPRODUCIBILITY.md) · "
    "[Kaggle and BigQuery integration](https://github.com/senanurcetin/ms-capital-market-forecasting/blob/main/docs/INTEGRATIONS.md) · "
    "[Model card](https://github.com/senanurcetin/ms-capital-market-forecasting/blob/main/MODEL_CARD.md)"
)

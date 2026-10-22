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

st.subheader("Warehouse verification snapshot")
warehouse_path = BUNDLED / "warehouse_audit.json"
if warehouse_path.is_file():
    warehouse = json.loads(warehouse_path.read_text(encoding="utf-8"))
    st.caption(f"Checked {warehouse['checked_at']} · bundled metadata snapshot · no live connection on this page")
    if warehouse["status"] == "pass":
        st.success(
            f"{warehouse['table_count']} tables inspected; {len(warehouse['contracts'])} metadata contracts passed. "
            f"All {warehouse['model_feature_count']} bundled model features are present with numeric types."
        )
    else:
        st.warning("Warehouse contracts failed. Inspect the snapshot before attempting a rebuild.")
    for warning in warehouse.get("warnings", []):
        st.warning(warning)
    st.dataframe(pd.DataFrame([
        {"Split": plan["split"], "Dry-run estimate (GB)": plan["estimated_bytes"] / 1e9
         if plan.get("estimated_bytes") is not None else None,
         "Within job ceiling": plan.get("within_job_ceiling"), "Executed": plan["executed"]}
        for plan in warehouse.get("download_plans", [])
    ]), hide_index=True, width="stretch")
    st.caption("Estimates above the job ceiling cannot run with the default client. No data query was executed for this audit.")
    for limitation in warehouse.get("limitations", []):
        st.caption(limitation)
    st.download_button("Download warehouse snapshot (JSON)", warehouse_path.read_bytes(),
                       file_name="mscapital-warehouse-snapshot.json", mime="application/json")
else:
    st.info("No bundled warehouse snapshot is available. Run the opt-in warehouse audit with owner credentials.")

st.subheader("Kaggle publication readback")
publication_path = BUNDLED / "kaggle_publications.json"
if publication_path.is_file():
    publications = json.loads(publication_path.read_text(encoding="utf-8"))
    st.caption(f"Checked {publications['checked_at']} · bundled readback snapshot · no Kaggle calls on this page")
    st.dataframe(pd.DataFrame([
        {"Notebook": row["id"], "Latest run": row.get("latest_run_status"),
         "Source matches repository": row.get("source_matches"), "Status": row["status"]}
        for row in publications["publications"]
    ]), hide_index=True, width="stretch")
    st.caption("Completed notebook execution re-renders historical analyses. It does not retrain the competition model or create a submission.")
    st.download_button("Download Kaggle readback (JSON)", publication_path.read_bytes(),
                       file_name="mscapital-kaggle-readback.json", mime="application/json")
else:
    st.info("No bundled publication readback is available.")

fixed_path = BUNDLED / "walkforward_fixed.json"
fixed_record = BUNDLED / "walkforward_fixed.run.json"
if fixed_path.is_file() and fixed_record.is_file():
    st.subheader("Retrospective full-data revalidation")
    fixed = json.loads(fixed_path.read_text(encoding="utf-8"))
    record = json.loads(fixed_record.read_text(encoding="utf-8"))
    st.caption(f"Recorded {record['created_at']} · {record['selected_rows']:,} source rows · {record['n_features']} features")
    st.dataframe(pd.DataFrame([
        {"Model": name, "Mean fold cosine": block["stability"]["mean"],
         "Scored folds": len(block["per_fold"]), "Protocol": block["evaluation_protocol"]}
        for name, block in fixed.items()
    ]), hide_index=True, width="stretch")
    st.info("Fixed-budget model fits receive no scored validation labels; blend weights use earlier folds only. Base models score five folds; the blend scores four. Compare on shared periods. These historical periods were already explored, so this is retrospective evidence, not a new independent test. No holdout remeasurement, model promotion or submission.")
    st.download_button("Download revalidation run record (JSON)", fixed_record.read_bytes(),
                       file_name="mscapital-revalidation-run.json", mime="application/json")

st.subheader("Reproduce the research")
st.markdown(
    "[Reproducibility guide](https://github.com/senanurcetin/ms-capital-market-forecasting/blob/main/docs/REPRODUCIBILITY.md) · "
    "[Kaggle and BigQuery integration](https://github.com/senanurcetin/ms-capital-market-forecasting/blob/main/docs/INTEGRATIONS.md) · "
    "[Model card](https://github.com/senanurcetin/ms-capital-market-forecasting/blob/main/MODEL_CARD.md)"
)

"""Page 3 - Prediction.

Two backends, and the fallback is the one that matters. Against a running FastAPI the page
exercises the real serving path, contract and all. Published, there is no API beside it, so
it loads the shipped artefact in-process instead - the same bundle the API would have
loaded. A page that could only say "cannot reach the API" would be dead on the one
deployment anybody actually sees.
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

import math

import streamlit as st

from streamlit_app.lib import (
    api_get,
    api_post,
    load_features,
    load_local_model,
    missing,
    page_header,
)

st.set_page_config(page_title="Predictions", layout="wide")
page_header("Prediction", "Pick a sample and ask the model")

health = api_get("/health")
local = None if health else load_local_model()

if health:
    if health.get("status") != "ok":
        st.warning(f"API degraded: {health.get('detail')}")
    info = api_get("/model-info") or {}
    metrics = info.get("metrics") or {}
    name, version = info.get("model_name", "-"), info.get("model_version", "-")
    n_features = info.get("n_features", "-")
    st.caption("Scoring through the FastAPI service.")
elif local is not None:
    b = local.bundle
    metrics = b.metrics or {}
    name, version, n_features = b.name, b.version, len(b.features)
    st.caption(
        "Scoring in-process with the shipped artefact - no API needed. Run `make api` to "
        "exercise the real serving path instead."
    )
else:
    st.error("No model available: neither the API nor a local artefact could be reached.")
    st.stop()

cosine = metrics.get("cosine")
c = st.columns(4)
c[0].metric("Model", name)
c[1].metric("Version", version)
c[2].metric("Features", n_features)
c[3].metric("Hold-out cosine", f"{cosine:+.4f}" if isinstance(cosine, int | float) else "-")

df = load_features(n_rows=5_000)
if df is None:
    missing("Feature set", "python -m src.features.assemble")
    st.stop()

sample = st.selectbox("sample_id", df["sample_id"].tolist()[:1000])
row = df[df["sample_id"] == sample].iloc[0]
features = {
    k: (0.0 if (isinstance(row[k], float) and math.isnan(row[k])) else float(row[k]))
    for k in df.columns
    if k not in ("sample_id", "month", "target")
}

if st.button("Predict", type="primary"):
    if health:
        status, body = api_post("/predict", {"features": features})
    else:
        value = float(local.predict([features])[0])
        status, body = 200, {
            "predicted_return": value,
            "direction": local.direction(value),
            "model_name": name, "model_version": version,
        }
    if status == 200 and body:
        a, b, d = st.columns(3)
        a.metric("Predicted return", f"{body['predicted_return'] * 1e4:+.2f} bps")
        b.metric("Direction", body["direction"])
        d.metric("Actual (train label)", f"{row['target'] * 1e4:+.2f} bps")
        st.caption(f"{body['model_name']} / {body['model_version']}")
        st.info(
            "Because cosine similarity is SCALE-INVARIANT, the prediction MAGNITUDE is "
            "not calibrated; only the sign and the ranking are meaningful."
        )
    else:
        st.error(f"API {status}: {body}")

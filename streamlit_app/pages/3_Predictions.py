"""Page 3 - Prediction."""
import math

import streamlit as st

from streamlit_app.lib import api_get, api_post, load_features, missing, page_header

st.set_page_config(page_title="Predictions", layout="wide")
page_header("Prediction", "Pick a sample and ask the model")

health = api_get("/health")
if not health:
    st.error("Cannot reach the API. Start it with `make api` or `docker compose up api`.")
    st.stop()
if health.get("status") != "ok":
    st.warning(f"API degraded: {health.get('detail')}")

info = api_get("/model-info") or {}
metrics = info.get("metrics") or {}
cosine = metrics.get("cosine")
c = st.columns(4)
c[0].metric("Model", info.get("model_name", "-"))
c[1].metric("Version", info.get("model_version", "-"))
c[2].metric("Features", info.get("n_features", "-"))
c[3].metric("Hold-out cosine", f"{cosine:+.4f}" if isinstance(cosine, (int, float)) else "-")

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
    status, body = api_post("/predict", {"features": features})
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

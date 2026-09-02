"""Sayfa 3 - Tahmin."""
import math

import streamlit as st

from streamlit_app.lib import api_get, api_post, load_features, missing, page_header

st.set_page_config(page_title="Predictions", layout="wide")
page_header("Tahmin", "Bir sample secin, modele sorun")

health = api_get("/health")
if not health:
    st.error("API'ye ulasilamiyor. `make api` veya `docker compose up api` ile baslatin.")
    st.stop()
if health.get("status") != "ok":
    st.warning(f"API degraded: {health.get('detail')}")

info = api_get("/model-info") or {}
metrics = info.get("metrics") or {}
cosine = metrics.get("cosine")
c = st.columns(4)
c[0].metric("Model", info.get("model_name", "-"))
c[1].metric("Versiyon", info.get("model_version", "-"))
c[2].metric("Feature", info.get("n_features", "-"))
c[3].metric("Egitim cosine", f"{cosine:+.4f}" if isinstance(cosine, (int, float)) else "-")

df = load_features(n_rows=5_000)
if df is None:
    missing("Feature seti", "python -m src.features.assemble")
    st.stop()

sample = st.selectbox("sample_id", df["sample_id"].tolist()[:1000])
row = df[df["sample_id"] == sample].iloc[0]
features = {
    k: (0.0 if (isinstance(row[k], float) and math.isnan(row[k])) else float(row[k]))
    for k in df.columns
    if k not in ("sample_id", "month", "target")
}

if st.button("Tahmin et", type="primary"):
    status, body = api_post("/predict", {"features": features})
    if status == 200 and body:
        a, b, d = st.columns(3)
        a.metric("Tahmin edilen getiri", f"{body['predicted_return'] * 1e4:+.2f} bps")
        b.metric("Yon", body["direction"])
        d.metric("Gercek (train etiketi)", f"{row['target'] * 1e4:+.2f} bps")
        st.caption(f"{body['model_name']} / {body['model_version']}")
        st.info(
            "Cosine similarity OLCEK-DEGISMEZ oldugu icin tahminin BUYUKLUGU kalibre "
            "degildir; anlamli olan isaret ve siralamadir."
        )
    else:
        st.error(f"API {status}: {body}")

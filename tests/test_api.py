"""API contract tests - driven by a fake model artefact (no training required)."""
import importlib
import json

import pytest
from fastapi.testclient import TestClient
from src.inference.predictor import Predictor, save_bundle

FEATURES = ["mkt_mid_last", "ord_ofi_60s", "txn_intensity_60s"]


class DummyModel:
    """predict() returns the difference of the first two features - deterministic."""

    def predict(self, X):
        return (X.iloc[:, 0] - X.iloc[:, 1]).to_numpy()


def _make_model_dir(tmp_path):
    import joblib

    d = tmp_path / "current"
    d.mkdir()
    joblib.dump(DummyModel(), d / "model.joblib")
    (d / "model_meta.json").write_text(
        json.dumps(
            {
                "name": "dummy",
                "version": "v0",
                "kind": "sklearn",
                "model_file": "model.joblib",
                "features": FEATURES,
                "metrics": {"cosine": 0.123},
                "trained_at": "2026-09-02T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    return d


def _client(monkeypatch, model_dir):
    monkeypatch.setenv("MSCAPITAL_MODEL_DIR", str(model_dir))
    import api.main as main

    importlib.reload(main)
    return TestClient(main.app)


@pytest.fixture
def client_with_model(tmp_path, monkeypatch):
    with _client(monkeypatch, _make_model_dir(tmp_path)) as c:
        yield c


@pytest.fixture
def client_without_model(tmp_path, monkeypatch):
    with _client(monkeypatch, tmp_path / "missing") as c:
        yield c


def test_health_ok_when_model_present(client_with_model):
    r = client_with_model.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_health_degraded_without_model_but_app_still_up(client_without_model):
    """The app must start even without a model so deployment order does not matter."""
    r = client_without_model.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "degraded" and r.json()["model_loaded"] is False


def test_predict_returns_503_without_model(client_without_model):
    r = client_without_model.post("/predict", json={"features": dict.fromkeys(FEATURES, 0.0)})
    assert r.status_code == 503


def test_model_info(client_with_model):
    body = client_with_model.get("/model-info").json()
    assert body["model_name"] == "dummy" and body["n_features"] == 3
    assert body["metrics"]["cosine"] == pytest.approx(0.123)


def test_predict_contract(client_with_model):
    r = client_with_model.post(
        "/predict",
        json={"features": {"mkt_mid_last": 1.0, "ord_ofi_60s": 0.4, "txn_intensity_60s": 2.0}},
    )
    assert r.status_code == 200
    b = r.json()
    assert b["predicted_return"] == pytest.approx(0.6)
    assert b["direction"] == "UP"
    assert set(b) == {"predicted_return", "direction", "model_name", "model_version"}


def test_predict_direction_down_and_flat(client_with_model):
    down = client_with_model.post(
        "/predict",
        json={"features": {"mkt_mid_last": 0.0, "ord_ofi_60s": 1.0, "txn_intensity_60s": 0.0}},
    ).json()
    flat = client_with_model.post(
        "/predict",
        json={"features": {"mkt_mid_last": 1.0, "ord_ofi_60s": 1.0, "txn_intensity_60s": 0.0}},
    ).json()
    assert down["direction"] == "DOWN" and flat["direction"] == "FLAT"


def test_missing_feature_is_rejected_not_silently_filled(client_with_model):
    """A missing feature must NOT be silently filled in - it must return 422."""
    r = client_with_model.post("/predict", json={"features": {"mkt_mid_last": 1.0}})
    assert r.status_code == 422 and "missing" in r.json()["detail"]


def test_batch_predict(client_with_model):
    rows = [
        {"mkt_mid_last": 1.0, "ord_ofi_60s": 0.0, "txn_intensity_60s": 0.0},
        {"mkt_mid_last": 0.0, "ord_ofi_60s": 2.0, "txn_intensity_60s": 0.0},
    ]
    b = client_with_model.post("/batch-predict", json={"rows": rows}).json()
    assert b["n"] == 2
    assert b["predictions"][0]["direction"] == "UP"
    assert b["predictions"][1]["predicted_return"] == pytest.approx(-2.0)


def test_batch_predict_rejects_empty(client_with_model):
    assert client_with_model.post("/batch-predict", json={"rows": []}).status_code == 422


def test_reload_picks_up_model(tmp_path, monkeypatch):
    """When a model appears later, /reload must pick it up without a restart."""
    model_dir = tmp_path / "current"
    with _client(monkeypatch, model_dir) as c:
        assert c.get("/health").json()["status"] == "degraded"
        _make_model_dir(tmp_path)
        assert c.post("/reload").json()["status"] == "ok"


def test_save_bundle_roundtrip(tmp_path):
    d = save_bundle(
        tmp_path / "m",
        model=DummyModel(),
        kind="sklearn",
        features=FEATURES,
        name="x",
        version="v1",
        metrics={"cosine": 0.5},
    )
    p = Predictor.from_dir(d)
    out = p.predict([{"mkt_mid_last": 2.0, "ord_ofi_60s": 0.5, "txn_intensity_60s": 9.0}])
    assert out[0] == pytest.approx(1.5)
    assert p.info()["model_version"] == "v1"


def test_extra_features_are_ignored_not_fatal(tmp_path):
    d = save_bundle(
        tmp_path / "m2", model=DummyModel(), kind="sklearn",
        features=FEATURES, name="x", version="v1",
    )
    p = Predictor.from_dir(d)
    row = {"mkt_mid_last": 3.0, "ord_ofi_60s": 1.0, "txn_intensity_60s": 0.0, "unknown_feature": 99.0}
    assert p.predict([row])[0] == pytest.approx(2.0)


# ------------------------------------------- serving a REAL ensemble over HTTP

"""Everything above drives a DummyModel through `kind="sklearn"`.

That covers the contract - status codes, missing features, batch shape - and covers none
of the artefact the API is actually pointed at. `ship.py` builds a three-model blend and
the serving layer was rewritten to load it without importing the training package; until
now that path was exercised only by calling Predictor directly, never through a request.
The defect it replaced lived exactly here: /model-info announced an ensemble while
/predict returned one model's numbers, and no test compared the two.
"""


@pytest.fixture(scope="module")
def trained_ensemble():
    """Three real models on a small synthetic frame - seconds, no credentials."""
    import numpy as np
    import pandas as pd
    from src.models.baseline import RidgeModel
    from src.models.lightgbm_model import LightGBMModel
    from src.models.xgboost_model import XGBoostModel

    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, len(FEATURES)))
    df = pd.DataFrame(X, columns=FEATURES)
    df.insert(0, "sample_id", np.arange(300))
    df["target"] = 0.5 * X[:, 0] - 0.2 * X[:, 1] + rng.normal(scale=0.4, size=300)

    models = {
        "lightgbm": LightGBMModel(num_boost_round=15, early_stopping_rounds=None),
        "xgboost": XGBoostModel(num_boost_round=15, early_stopping_rounds=None),
        "ridge": RidgeModel(alpha=1.0),
    }
    for m in models.values():
        m.fit(df, df["target"].to_numpy())
    return models, {"lightgbm": 0.6, "xgboost": 0.3, "ridge": 0.1}, df


@pytest.fixture
def ensemble_client(tmp_path, monkeypatch, trained_ensemble):
    models, weights, _ = trained_ensemble
    d = tmp_path / "shipped"
    save_bundle(d, model={"models": models, "weights": weights}, kind="ensemble",
                features=FEATURES, name="ensemble", version="v4",
                metrics={"blend_in_sample_cosine": 0.1376})
    with _client(monkeypatch, d) as c:
        yield c


def test_the_api_reports_the_ensemble_it_is_actually_serving(ensemble_client):
    body = ensemble_client.get("/model-info").json()
    assert body["model_name"] == "ensemble"
    assert body["n_features"] == len(FEATURES)


def test_predict_over_http_returns_the_blend_not_one_member(ensemble_client,
                                                            trained_ensemble):
    """The exact confusion that shipped: the endpoint must not answer with LightGBM."""
    import numpy as np

    models, weights, df = trained_ensemble
    row = df[FEATURES].iloc[0]
    served = ensemble_client.post(
        "/predict", json={"features": row.to_dict()}).json()["predicted_return"]

    frame = row.to_frame().T.astype("float32")
    blend = float(sum(w * models[n].predict(frame)[0] for n, w in weights.items()))
    lightgbm_only = float(models["lightgbm"].predict(frame)[0])

    assert served == pytest.approx(blend, rel=1e-3)
    assert not np.isclose(served, lightgbm_only, rtol=1e-3), (
        "the endpoint is answering with a single member of the blend"
    )


def test_batch_predict_over_http_matches_the_blend(ensemble_client, trained_ensemble):
    models, weights, df = trained_ensemble
    rows = df[FEATURES].head(4)
    body = ensemble_client.post("/batch-predict",
                                json={"rows": rows.to_dict("records")})
    assert body.status_code == 200
    got = [p["predicted_return"] for p in body.json()["predictions"]]

    frame = rows.astype("float32")
    expected = sum(w * models[n].predict(frame) for n, w in weights.items())
    assert got == pytest.approx(list(expected), rel=1e-3)


def test_reload_swaps_a_single_model_for_the_ensemble(tmp_path, monkeypatch,
                                                      trained_ensemble):
    """The upgrade path: an API already serving the LightGBM must pick up the blend.

    Worth its own test because the two artefacts have different FILES, not just different
    contents - swapping formats in place is where a stale model.txt could be left behind
    and quietly reloaded.
    """
    models, weights, _ = trained_ensemble
    d = tmp_path / "current"
    save_bundle(d, model=models["lightgbm"].booster_, kind="lightgbm",
                features=FEATURES, name="lightgbm", version="v3")

    with _client(monkeypatch, d) as client:
        assert client.get("/model-info").json()["model_name"] == "lightgbm"

        save_bundle(d, model={"models": models, "weights": weights}, kind="ensemble",
                    features=FEATURES, name="ensemble", version="v4")
        assert client.post("/reload").status_code == 200
        assert client.get("/model-info").json()["model_name"] == "ensemble"

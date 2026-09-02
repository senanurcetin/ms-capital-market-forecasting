"""API sozlesme testleri - sahte bir model artefakti ile (egitim gerektirmez)."""
import importlib
import json

import pytest
from fastapi.testclient import TestClient

from src.inference.predictor import Predictor, save_bundle

FEATURES = ["mkt_mid_last", "ord_ofi_60s", "txn_intensity_60s"]


class DummyModel:
    """predict() ilk iki feature'in farkini doner - deterministik ve bagimliliksiz."""

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
    with _client(monkeypatch, tmp_path / "yok") as c:
        yield c


def test_health_ok_when_model_present(client_with_model):
    r = client_with_model.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_health_degraded_without_model_but_app_still_up(client_without_model):
    """Model yokken bile uygulama ayaga kalkmali - deploy sirasi bagimsiz olsun."""
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
    """Eksik feature SESSIZCE doldurulmamali - acik 422 donmeli."""
    r = client_with_model.post("/predict", json={"features": {"mkt_mid_last": 1.0}})
    assert r.status_code == 422 and "eksik" in r.json()["detail"]


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
    """Model sonradan yazildiginda /reload servisi yeniden baslatmadan yuklemeli."""
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
    row = {"mkt_mid_last": 3.0, "ord_ofi_60s": 1.0, "txn_intensity_60s": 0.0, "bilinmeyen": 99.0}
    assert p.predict([row])[0] == pytest.approx(2.0)

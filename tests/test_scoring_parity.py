"""The Prediction page's two scoring paths must agree (issue #93).

The page scores through the FastAPI service when one is running and in-process with the
shipped artefact otherwise. The public app always takes the in-process path, so a silent
difference between the two would never be seen. Both paths here load the same shipped
artefact in `results/`; no network and no training data are needed.
"""
import importlib
import random
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from src.inference.predictor import Predictor, load_bundle
from streamlit_app.lib import score_in_process

SHIPPED = Path(__file__).resolve().parents[1] / "results"


def _client(monkeypatch, deadband: str | None):
    monkeypatch.setenv("MSCAPITAL_MODEL_DIR", str(SHIPPED))
    if deadband is None:
        monkeypatch.delenv("MSCAPITAL_DIRECTION_DEADBAND", raising=False)
    else:
        monkeypatch.setenv("MSCAPITAL_DIRECTION_DEADBAND", deadband)
    import api.main as main

    importlib.reload(main)
    return TestClient(main.app)


@pytest.fixture
def api_client(monkeypatch):
    monkeypatch.setenv("MSCAPITAL_MODEL_DIR", str(SHIPPED))
    monkeypatch.delenv("MSCAPITAL_DIRECTION_DEADBAND", raising=False)
    import api.main as main

    importlib.reload(main)
    with TestClient(main.app) as client:
        yield client


@pytest.fixture(scope="module")
def local() -> Predictor:
    return Predictor(load_bundle(SHIPPED))


def _rows(features: list[str], n: int = 25) -> list[dict]:
    """Deterministic rows spread wide enough to hit UP, DOWN and FLAT-adjacent outputs."""
    rng = random.Random(93)
    rows = [dict.fromkeys(features, 0.0)]
    for _ in range(n - 1):
        rows.append({k: rng.gauss(0.0, 2.0) for k in features})
    return rows


def test_shipped_artefact_is_present():
    assert (SHIPPED / "model.txt").is_file() and (SHIPPED / "model_meta.json").is_file()


def test_api_and_in_process_scoring_agree_on_the_shipped_artefact(api_client, local):
    assert api_client.get("/health").json()["status"] == "ok"
    directions = set()
    for features in _rows(list(local.bundle.features)):
        r = api_client.post("/predict", json={"features": features})
        assert r.status_code == 200, r.text
        api, page = r.json(), score_in_process(local, features)
        assert api["predicted_return"] == pytest.approx(page["predicted_return"], rel=0, abs=1e-12)
        assert api["direction"] == page["direction"]
        directions.add(page["direction"])
    # Guard against a degenerate sample where every row lands on one side.
    assert {"UP", "DOWN"} <= directions


def test_both_paths_report_the_same_model(api_client, local):
    info = api_client.get("/model-info").json()
    assert info["model_name"] == local.bundle.name
    assert info["model_version"] == local.bundle.version
    assert info["n_features"] == len(local.bundle.features)


def test_both_paths_apply_the_configured_direction_deadband(monkeypatch, local):
    """A deadband set for the API must label the page's in-process result the same way."""
    rows = _rows(list(local.bundle.features))
    values = sorted(abs(score_in_process(local, f)["predicted_return"]) for f in rows)
    deadband = values[len(values) // 2]   # half the rows fall inside it, half outside
    with _client(monkeypatch, repr(deadband)) as client:
        seen = set()
        for features in rows:
            api = client.post("/predict", json={"features": features}).json()
            page = score_in_process(local, features)
            assert api["direction"] == page["direction"]
            seen.add(page["direction"])
    assert "FLAT" in seen

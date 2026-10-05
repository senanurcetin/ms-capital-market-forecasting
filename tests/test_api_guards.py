"""Request guards and /metrics: limits must bite, and /health must never be limited."""
import importlib

import pytest
from api.guards import Metrics, RateLimiter
from fastapi.testclient import TestClient

from tests.test_api import FEATURES, _make_model_dir

ROW = dict.fromkeys(FEATURES, 1.0)


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def _client(monkeypatch, tmp_path, **env):
    monkeypatch.setenv("MSCAPITAL_MODEL_DIR", str(_make_model_dir(tmp_path)))
    for key, value in env.items():
        monkeypatch.setenv(key, str(value))
    import api.main as main

    importlib.reload(main)
    return TestClient(main.app)


# --- RateLimiter ---------------------------------------------------------------------------

def test_limiter_allows_up_to_the_limit_then_reports_a_wait():
    clock = FakeClock()
    lim = RateLimiter(2, clock=clock)
    assert lim.check("a") is None and lim.check("a") is None
    wait = lim.check("a")
    assert wait is not None and 0 < wait <= 60


def test_limiter_recovers_after_the_window_and_isolates_clients():
    clock = FakeClock()
    lim = RateLimiter(1, clock=clock)
    assert lim.check("a") is None
    assert lim.check("a") is not None
    assert lim.check("b") is None          # another client has its own budget
    clock.t = 61
    assert lim.check("a") is None


def test_limiter_disabled_at_zero():
    lim = RateLimiter(0)
    assert all(lim.check("a") is None for _ in range(1000))


def test_limiter_memory_is_bounded(monkeypatch):
    clock = FakeClock()
    lim = RateLimiter(5, clock=clock)
    monkeypatch.setattr(RateLimiter, "MAX_KEYS", 10)
    for i in range(10):
        lim.check(f"k{i}")
    clock.t = 120                           # every window has expired
    lim.check("fresh")
    assert len(lim._hits) <= 10


# --- over HTTP -----------------------------------------------------------------------------

def test_rate_limit_returns_429_with_retry_after(monkeypatch, tmp_path):
    with _client(monkeypatch, tmp_path, MSCAPITAL_RATE_LIMIT_PER_MIN=2) as c:
        body = {"features": ROW}
        assert c.post("/predict", json=body).status_code == 200
        assert c.post("/predict", json=body).status_code == 200
        r = c.post("/predict", json=body)
        assert r.status_code == 429 and int(r.headers["Retry-After"]) >= 1


def test_health_is_not_rate_limited(monkeypatch, tmp_path):
    with _client(monkeypatch, tmp_path, MSCAPITAL_RATE_LIMIT_PER_MIN=1) as c:
        assert all(c.get("/health").status_code == 200 for _ in range(5))


def test_no_limit_by_default(monkeypatch, tmp_path):
    monkeypatch.delenv("MSCAPITAL_RATE_LIMIT_PER_MIN", raising=False)
    with _client(monkeypatch, tmp_path) as c:
        assert all(c.post("/predict", json={"features": ROW}).status_code == 200
                   for _ in range(20))


def test_oversized_body_is_rejected_before_parsing(monkeypatch, tmp_path):
    with _client(monkeypatch, tmp_path, MSCAPITAL_MAX_BODY_BYTES=100) as c:
        r = c.post("/batch-predict", json={"rows": [ROW] * 50})
        assert r.status_code == 413


def test_batch_row_cap_is_configurable(monkeypatch, tmp_path):
    with _client(monkeypatch, tmp_path, MSCAPITAL_MAX_BATCH_ROWS=3) as c:
        assert c.post("/batch-predict", json={"rows": [ROW] * 3}).status_code == 200
        assert c.post("/batch-predict", json={"rows": [ROW] * 4}).status_code == 422


# --- metrics -------------------------------------------------------------------------------

def test_metrics_count_requests_by_route_template_and_rows(monkeypatch, tmp_path):
    with _client(monkeypatch, tmp_path) as c:
        c.post("/predict", json={"features": ROW})
        c.post("/batch-predict", json={"rows": [ROW] * 4})
        c.get("/nonexistent")
        text = c.get("/metrics").text
    assert 'mscapital_http_requests_total{method="POST",path="/predict",status="200"} 1' in text
    assert 'path="/batch-predict",status="200"} 1' in text
    assert "mscapital_rows_scored_total 5" in text
    # Unknown paths share one label, so a scanner cannot blow up the metric cardinality.
    assert 'path="unmatched",status="404"} 1' in text
    assert "/nonexistent" not in text


def test_metrics_never_contain_request_contents(monkeypatch, tmp_path):
    with _client(monkeypatch, tmp_path) as c:
        c.post("/predict", json={"features": {**ROW, "mkt_mid_last": 123456.789}})
        assert "123456" not in c.get("/metrics").text


def test_metrics_render_is_valid_exposition_format():
    m = Metrics()
    m.observe("GET", "/health", 200, 0.012)
    for line in m.render().strip().splitlines():
        assert line.startswith("#") or len(line.rsplit(" ", 1)) == 2
        if not line.startswith("#"):
            float(line.rsplit(" ", 1)[1])


@pytest.mark.parametrize("kind", ["predict", "batch"])
def test_failed_scoring_does_not_count_rows(monkeypatch, tmp_path, kind):
    with _client(monkeypatch, tmp_path) as c:
        if kind == "predict":
            c.post("/predict", json={"features": {"mkt_mid_last": 1.0}})
        else:
            c.post("/batch-predict", json={"rows": [{"mkt_mid_last": 1.0}]})
        assert "mscapital_rows_scored_total 0" in c.get("/metrics").text

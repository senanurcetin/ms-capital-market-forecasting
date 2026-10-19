"""Evidence corruption must be visible, and a refresh must change the inventory hash."""
import json
import shutil
from pathlib import Path

import pytest
from src.data.query_budget import DEFAULT_MAX_BYTES, query_config
from src.evaluation.evidence import audit

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def bundle(tmp_path):
    for name in ("walkforward_summary.json", "walkforward_summary.csv", "holdout_metrics.json", "leaderboard.json", "model_meta.json"):
        shutil.copyfile(ROOT / "results" / name, tmp_path / name)
    return tmp_path


def test_committed_evidence_reconciles():
    report = audit(ROOT / "results")
    assert report["status"] == "pass", report["errors"]
    assert report["fold_count"] == 5
    assert report["served_model"]["version"] == "v3"


def test_csv_metric_corruption_fails(bundle):
    path = bundle / "walkforward_summary.csv"
    path.write_text("model,cosine_mean\nensemble,0.99\n", encoding="utf-8")
    report = audit(bundle)
    assert report["status"] == "fail"
    assert "CSV and JSON walk-forward means disagree" in report["errors"]


@pytest.mark.parametrize("value", [None, True, "0.129", float("nan"), float("inf")])
def test_invalid_leaderboard_is_never_a_measurement(bundle, value):
    path = bundle / "leaderboard.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["our_score"] = value
    path.write_text(json.dumps(data), encoding="utf-8")
    report = audit(bundle)
    assert report["status"] == "fail"
    assert report["metrics"]["leaderboard_cosine"] is None
    json.dumps(report, allow_nan=False)


def test_missing_file_fails_and_file_changes_update_hash(bundle):
    before = audit(bundle)
    (bundle / "leaderboard.json").unlink()
    after = audit(bundle)
    assert after["status"] == "fail"
    assert before["files"] != after["files"]


def test_default_scan_limit(monkeypatch):
    monkeypatch.delenv("MSCAPITAL_BQ_MAX_BYTES", raising=False)
    assert query_config().maximum_bytes_billed == DEFAULT_MAX_BYTES


@pytest.mark.parametrize("value", ["0", "-1", "unlimited", "1.5", ""])
def test_unlimited_or_invalid_scan_budget_is_rejected(monkeypatch, value):
    monkeypatch.setenv("MSCAPITAL_BQ_MAX_BYTES", value)
    with pytest.raises(ValueError, match="positive integer"):
        query_config()


def test_explicit_scan_budget(monkeypatch):
    monkeypatch.setenv("MSCAPITAL_BQ_MAX_BYTES", "10000000")
    assert query_config().maximum_bytes_billed == 10_000_000


def test_missing_directory_is_an_explicit_failed_audit(tmp_path):
    report = audit(tmp_path / "missing")
    assert report["status"] == "fail"
    assert report["files"] == []


def test_malformed_fold_structure_fails_without_crashing(bundle):
    path = bundle / "walkforward_summary.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["ensemble"]["per_fold"] = [None]
    path.write_text(json.dumps(data), encoding="utf-8")
    assert audit(bundle)["status"] == "fail"

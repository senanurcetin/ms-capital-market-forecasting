"""Regression tests for blend leakage, aligned periods and portable experiment records."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from src.evaluation.uncertainty import block_indices, compare
from src.models.ensemble import evaluate_frozen_ensemble
from src.models.run_record import fingerprint, record_run, write_record
from src.models.train import run_walk_forward

ROOT = Path(__file__).resolve().parents[1]


def test_current_targets_cannot_change_blend_weights():
    rng = np.random.default_rng(7)
    truth = rng.normal(size=100)
    earlier = {"a": truth.copy(), "b": -truth}
    current = {"a": truth.copy(), "b": -truth}
    a = evaluate_frozen_ensemble(current, truth, earlier, truth)
    b = evaluate_frozen_ensemble(current, -truth, earlier, truth)
    assert a["weights"] == b["weights"]
    assert a["ensemble_score"] == pytest.approx(1)
    assert b["ensemble_score"] == pytest.approx(-1)


def test_training_preserves_fold_ids_and_uses_only_earlier_calibration():
    from src.models.baseline import MeanPredictor, ZeroPredictor

    months = np.repeat(np.arange(71), 3)
    frame = pd.DataFrame({"sample_id": np.arange(len(months)), "month": months,
                          "target": np.sin(months) * 0.001, "mkt_x": np.cos(months)})
    result = run_walk_forward({"zero": ZeroPredictor, "mean": MeanPredictor}, frame,
                              log_mlflow=False, last_folds=2)
    assert [row["fold"] for row in result["mean"]["per_fold"]] == [4, 5]
    ensemble = result["ensemble"]["per_fold"]
    assert len(ensemble) == 1 and ensemble[0]["fold"] == 5
    assert ensemble[0]["calibration_folds"] == [4]
    assert result["ensemble"]["evaluation_protocol"] == "prior_fold_blend"
    assert result["ensemble"]["stability"]["worst_fold"] == 5
    assert result["mean"]["stability"]["worst_fold"] in [4, 5]


def test_fixed_fit_never_receives_scored_validation_labels():
    class FixedModel:
        name = "fixed"

        def fit(self, X, y, **kwargs):
            assert kwargs == {}, "Outer validation must not be passed to fitting"
            self.train_last_month = X.month.max()
            return self

        def predict(self, X):
            assert X.month.min() > self.train_last_month
            return np.zeros(len(X))

    months = np.repeat(np.arange(71), 3)
    frame = pd.DataFrame({"sample_id": np.arange(len(months)), "month": months,
                          "target": np.sin(months) * 0.001, "mkt_x": np.cos(months)})
    result = run_walk_forward({"fixed": FixedModel}, frame, log_mlflow=False, fit_validation=False)
    assert result["fixed"]["evaluation_protocol"] == "walkforward_fixed_fit"
    assert len(result["fixed"]["per_fold"]) == 5


def test_bootstrap_preserves_adjacency_and_seed():
    a = block_indices(5, 2, 100, 42)
    np.testing.assert_array_equal(a, block_indices(5, 2, 100, 42))
    assert ((a[:, 1] - a[:, 0]) % 5 == 1).all()
    assert ((a[:, 3] - a[:, 2]) % 5 == 1).all()


def test_historical_blend_is_excluded_and_pairs_have_identical_periods():
    summary = json.loads((ROOT / "results/walkforward_summary.json").read_text())
    report = compare(summary, draws=100)
    assert report["excluded"]
    assert {r["model"] for r in report["comparisons"]} == set(summary) - {"ensemble"}
    for row in report["comparisons"]:
        if row["model"] == "lightgbm":
            assert row["delta_low"] == row["delta_high"] == 0


def test_fixed_fit_uncertainty_uses_the_actual_fit_protocol():
    summary = {
        name: {"evaluation_protocol": "walkforward_fixed_fit",
               "per_fold": [{"fold": i, "cosine": 0.01 * i} for i in range(1, 6)]}
        for name in ("lightgbm", "ridge")
    }
    report = compare(summary, draws=100)
    assert any("fixed fits without scored validation labels" in text for text in report["limits"])
    assert not any("early stopping used" in text for text in report["limits"])


@pytest.mark.parametrize("bad", [None, True, float("nan"), float("inf"), 2.0])
def test_bad_scores_fail_closed(bad):
    summary = json.loads((ROOT / "results/walkforward_summary.json").read_text())
    summary["ridge"]["per_fold"][0]["cosine"] = bad
    with pytest.raises(ValueError, match="invalid cosine"):
        compare(summary, draws=100)


def test_manifest_records_selected_data_parameters_and_output(tmp_path):
    source = tmp_path / "input.parquet"
    source.write_bytes(b"fixture")
    output = tmp_path / "summary.json"
    output.write_text('{"ok": true}')
    frame = pd.DataFrame({"month": [0, 1], "target": [0., 1.], "mkt_x": [1., 2.]})
    record = record_run(frame=frame, source=fingerprint(source), config={"embargo_months": 1},
                        options={"sampling_seed": 42}, models={"ridge": {"alpha": 10}},
                        summary_path=output, root=ROOT)
    path = tmp_path / "summary.run.json"
    write_record(path, record)
    saved = json.loads(path.read_text())
    assert saved["n_features"] == 1 and saved["selected_rows"] == 2
    assert saved["summary"]["sha256"] == fingerprint(output)["sha256"]
    assert saved["source"]["file"] == "input.parquet"
    assert "credentials" not in saved


def test_training_cli_writes_protocol_and_run_sidecar(tmp_path, monkeypatch):
    from src.config import Config
    from src.models import train

    months = np.repeat(np.arange(71), 3)
    frame = pd.DataFrame({"sample_id": np.arange(len(months)), "month": months,
                          "target": np.sin(months) * 0.001, "mkt_x": np.cos(months)})
    frame.to_parquet(tmp_path / "dataset_train.parquet", index=False)
    config = train.load_config()
    config = Config({**config, "paths": {**config.paths, "features": str(tmp_path)}})
    monkeypatch.setattr(train, "load_config", lambda: config)
    output = tmp_path / "new-run/summary.json"
    train.main(["--models", "zero,mean", "--no-mlflow", "--folds", "2", "--out", str(output)])
    summary = json.loads(output.read_text())
    record = json.loads(output.with_suffix(".run.json").read_text())
    assert summary["ensemble"]["evaluation_protocol"] == "prior_fold_blend"
    assert [f["fold"] for f in summary["ensemble"]["per_fold"]] == [5]
    assert record["selected_rows"] == len(frame)
    assert record["options"]["last_folds"] == 2

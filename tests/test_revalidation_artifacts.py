"""Full-data evidence must reconcile with its immutable runner record and fold masks."""
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_run_record_identifies_the_exact_published_summary_and_full_population():
    path = ROOT / "results/walkforward_fixed.json"
    record = json.loads(path.with_suffix(".run.json").read_text())
    assert record["summary"]["file"] == path.name
    assert record["summary"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert record["summary"]["bytes"] == path.stat().st_size
    assert record["selected_rows"] == sum(record["rows_per_month"].values()) == 1_257_637
    assert record["n_features"] == 292
    assert record["code"]["dirty"] is False
    assert record["options"]["sampling"] == "none"
    assert record["options"]["fit_validation"] is False


def test_fold_rows_match_the_full_manifest_and_holdout_is_never_scored():
    report = json.loads((ROOT / "results/walkforward_fixed.json").read_text())
    record = json.loads((ROOT / "results/walkforward_fixed.run.json").read_text())
    counts = {int(month): count for month, count in record["rows_per_month"].items()}
    for name in ("lightgbm", "ridge"):
        assert report[name]["evaluation_protocol"] == "walkforward_fixed_fit"
        assert [row["fold"] for row in report[name]["per_fold"]] == list(range(1, 6))
        for row in report[name]["per_fold"]:
            assert row["val_months"][1] < 65
            assert row["train_rows"] == sum(n for m, n in counts.items() if row["train_months"][0] <= m <= row["train_months"][1])
            assert row["val_rows"] == sum(n for m, n in counts.items() if row["val_months"][0] <= m <= row["val_months"][1])
    assert report["ensemble"]["evaluation_protocol"] == "prior_fold_blend"
    assert [row["fold"] for row in report["ensemble"]["per_fold"]] == [2, 3, 4, 5]
    for row in report["ensemble"]["per_fold"]:
        assert row["calibration_folds"] == list(range(1, row["fold"]))


def test_csv_means_and_shared_period_comparisons_reconcile():
    report = json.loads((ROOT / "results/walkforward_fixed.json").read_text())
    with (ROOT / "results/walkforward_fixed.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            assert math.isclose(float(row["cosine_mean"]), report[row["model"]]["stability"]["mean"], abs_tol=1e-12)
    uncertainty = json.loads((ROOT / "results/walkforward_fixed.uncertainty.json").read_text())
    assert uncertainty["folds"] == [2, 3, 4, 5]
    assert uncertainty["source_sha256"] == hashlib.sha256((ROOT / "results/walkforward_fixed.json").read_bytes()).hexdigest()
    for comparison in uncertainty["comparisons"]:
        model = report[comparison["model"]]
        values = [row.get("ensemble_score", row.get("cosine")) for row in model["per_fold"] if row["fold"] in uncertainty["folds"]]
        assert math.isclose(comparison["mean_fold_cosine"], sum(values) / 4, abs_tol=1e-12)

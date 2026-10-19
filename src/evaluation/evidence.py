"""Offline integrity audit of the published research bundle; no cloud credentials needed."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def audit(results: Path) -> dict[str, Any]:
    """Inventory bytes and reconcile headline metrics against their independent exports.

    Hashes identify the files read, not their scientific validity or their creation date.
    Malformed/missing evidence fails visibly rather than manufacturing a zero score.
    """
    errors: list[str] = []
    inventory = []
    for path in sorted(results.iterdir()) if results.is_dir() else []:
        if path.is_file():
            inventory.append({
                "file": path.name, "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            })

    def read(name: str) -> dict:
        try:
            value = json.loads((results / name).read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ValueError("expected a JSON object")
            return value
        except (OSError, ValueError) as exc:
            errors.append(f"{name}: {exc}")
            return {}

    def score(value: object, label: str) -> float | None:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            errors.append(f"{label}: missing or non-finite numeric score")
            return None
        return float(value)

    summary = read("walkforward_summary.json")
    def mapping(value: object) -> dict:
        return value if isinstance(value, dict) else {}

    ensemble = mapping(summary.get("ensemble"))
    cv = score(mapping(ensemble.get("stability")).get("mean"), "Walk-forward mean")
    folds = ensemble.get("per_fold", [])
    if not isinstance(folds, list) or any(not isinstance(f, dict) for f in folds):
        errors.append("Invalid walk-forward fold records")
        folds = []
    fold_scores = [score(f.get("ensemble_score", f.get("cosine")), "Fold score") for f in folds]
    if not fold_scores:
        errors.append("No walk-forward folds recorded")
    elif (
        cv is not None
        and all(v is not None for v in fold_scores)
        and not math.isclose(
            sum(v for v in fold_scores if v is not None) / len(fold_scores), cv, abs_tol=1e-10
        )
    ):
        errors.append("Walk-forward mean disagrees with per-fold scores")
    try:
        with (results / "walkforward_summary.csv").open(encoding="utf-8", newline="") as handle:
            rows = [r for r in csv.DictReader(handle) if r.get("model") == "ensemble"]
        if len(rows) != 1 or cv is None or not math.isclose(float(rows[0]["cosine_mean"]), cv, abs_tol=1e-10):
            errors.append("CSV and JSON walk-forward means disagree")
    except (OSError, ValueError, KeyError) as exc:
        errors.append(f"walkforward_summary.csv: {exc}")
    holdout = read("holdout_metrics.json")
    ho = score(mapping(holdout.get("scores")).get("cosine"), "Hold-out cosine")
    leaderboard = read("leaderboard.json")
    lb = score(leaderboard.get("our_score"), "Leaderboard cosine")
    if not leaderboard.get("captured"):
        errors.append("Leaderboard capture date is missing")
    model = read("model_meta.json")
    if not model.get("version"):
        errors.append("Served model version is missing")
    return {
        "schema_version": 1, "status": "pass" if not errors else "fail",
        "scope": "Committed research snapshot; no live Kaggle or BigQuery verification",
        "metrics": {"walkforward_cosine": cv, "holdout_cosine": ho, "leaderboard_cosine": lb},
        "fold_count": len(folds), "leaderboard_captured": leaderboard.get("captured"),
        "served_model": {"name": model.get("name"), "version": model.get("version")},
        "files": inventory, "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--output", type=Path, help="Optional JSON report outside results/")
    args = parser.parse_args()
    if args.output and args.output.resolve().is_relative_to(args.results.resolve()):
        parser.error("Report must be outside results/ to avoid hashing itself")
    report = audit(args.results)
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(f"Evidence audit: {report['status']} ({len(report['files'])} files)")
    for error in report["errors"]:
        print(f"  {error}")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

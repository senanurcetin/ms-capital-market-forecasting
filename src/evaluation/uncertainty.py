"""Aligned fold comparisons with exploratory circular block-bootstrap intervals."""
from __future__ import annotations

import argparse
import hashlib
import json
from math import isfinite
from numbers import Real
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def block_indices(n: int, block: int, draws: int, seed: int) -> np.ndarray:
    """Resample adjacent circular blocks; one shared index matrix preserves pairing."""
    if not 1 <= block < n or not 100 <= draws <= 100_000 or n > 1000:
        raise ValueError("Need 1 <= block < periods <= 1000 and 100 <= draws <= 100000")
    starts = np.random.default_rng(seed).integers(0, n, size=(draws, (n + block - 1) // block))
    return ((starts[:, :, None] + np.arange(block)) % n).reshape(draws, -1)[:, :n]


def compare(summary: dict, *, reference: str = "lightgbm", draws: int = 10_000, seed: int = 42) -> dict:
    models: dict[str, dict[int, float]] = {}
    excluded = []
    for name, record in summary.items():
        if name == "ensemble" and record.get("evaluation_protocol") != "prior_fold_blend":
            excluded.append("ensemble: historical weights were fitted and scored on the same fold")
            continue
        scores_by_fold = {}
        for row in record.get("per_fold", []):
            fold = row.get("fold")
            score = row.get("ensemble_score", row.get("cosine"))
            if type(fold) is not int or fold < 1 or fold in scores_by_fold:
                raise ValueError(f"{name}: missing, duplicate or invalid fold identity")
            if isinstance(score, bool) or not isinstance(score, Real) or not isfinite(float(score)) or abs(float(score)) > 1 + 1e-12:
                raise ValueError(f"{name}: invalid cosine score")
            scores_by_fold[fold] = float(score)
        if not scores_by_fold:
            raise ValueError(f"{name}: no fold scores")
        models[name] = scores_by_fold
    if reference not in models or len(models) < 2:
        raise ValueError("Reference and at least one comparison model are required")
    folds = sorted(set.intersection(*(set(v) for v in models.values())))
    if len(folds) < 3 or any(b - a != 1 for a, b in zip(folds, folds[1:], strict=False)):
        raise ValueError("Need at least three shared consecutive ordered folds")
    rows = []
    for block in (1, 2):
        indices = block_indices(len(folds), block, draws, seed)
        base = np.array([models[reference][f] for f in folds])
        for name in sorted(models):
            values = np.array([models[name][f] for f in folds])
            delta = values - base
            low, high = np.quantile(values[indices].mean(axis=1), [0.025, 0.975])
            delta_low, delta_high = np.quantile(delta[indices].mean(axis=1), [0.025, 0.975])
            rows.append({
                "model": name, "block_length": block, "mean_fold_cosine": float(values.mean()),
                "interval_low": float(low), "interval_high": float(high),
                "paired_delta": float(delta.mean()), "delta_low": float(delta_low), "delta_high": float(delta_high),
                "wins_vs_reference": int((delta > 0).sum()),
            })
    return {
        "schema_version": 1, "reference": reference, "folds": folds,
        "method": "circular block bootstrap, percentile 95%; shared paired indices",
        "estimand": "unweighted mean of recorded fold cosine scores, NOT pooled-row cosine",
        "draws": draws, "seed": seed, "excluded": excluded, "comparisons": rows,
        "limits": [
            "Exploratory resampling intervals, not calibrated generalization confidence bounds.",
            "Few folds, overlapping expanding training sets and nonstationarity limit inference.",
            "Block 1 is an IID sensitivity check; block 2 preserves local adjacency but wraps endpoints.",
            ("Base models used fixed fits without scored validation labels; these remain retrospective historical folds."
             if all(block.get("evaluation_protocol") == "walkforward_fixed_fit"
                    for name, block in summary.items() if name != "ensemble")
             else "Base-model early stopping used each validation fold; this is not nested outer validation."),
            "No significance tests, multiple-comparison claims or new competition scores are produced.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "results/walkforward_summary.json")
    parser.add_argument("--output", type=Path, default=ROOT / "site/uncertainty.json")
    args = parser.parse_args()
    raw = args.source.read_bytes()
    report = compare(json.loads(raw))
    report["source_sha256"] = hashlib.sha256(raw).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Compared {len(report['folds'])} aligned folds; wrote {args.output}")


if __name__ == "__main__":
    main()

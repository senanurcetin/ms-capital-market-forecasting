"""Train vs test distribution drift, measured feature by feature.

WHY THIS MATTERS HERE: the raw event density already differs between the splits -
test carries ~36% more order events per sample (135.2 -> 184.4). That was the reason
the feature layer emits rates and ratios rather than raw counts. This module checks
whether that design choice actually worked, and flags any feature that still shifts.

The metric is a standardised mean difference (an effect size), not a p-value:

    shift = |mean_test - mean_train| / std_train

With 1.26M and 648k samples, any test of "are these distributions identical" returns
a vanishing p-value for a difference far too small to matter. Effect size answers the
question that is actually useful: is the shift large relative to the feature's own
spread? Rules of thumb: < 0.1 negligible, 0.1-0.3 small, 0.3-0.5 moderate, > 0.5 large.

Null-rate drift is reported separately: a feature that is 5% NULL in train and 40%
NULL in test is broken for the model even if the non-null values line up.
"""
from __future__ import annotations

import argparse
import logging

import pandas as pd
from google.cloud import bigquery

from src.config import load_config
from src.data.bq_loader import client
from src.models.base import NON_FEATURES

log = logging.getLogger(__name__)

STAT_SEP = "__"


def _stats_sql(table: str, features: list[str]) -> str:
    parts = []
    for f in features:
        parts.append(f"AVG(`{f}`) AS `{f}{STAT_SEP}mean`")
        parts.append(f"STDDEV(`{f}`) AS `{f}{STAT_SEP}std`")
        parts.append(f"COUNTIF(`{f}` IS NULL) / COUNT(*) AS `{f}{STAT_SEP}null`")
    return "SELECT\n  " + ",\n  ".join(parts) + f"\nFROM `{table}`"


def _feature_list(bq: bigquery.Client, table: str) -> list[str]:
    schema = bq.get_table(table).schema
    return [f.name for f in schema if f.name not in NON_FEATURES]


def _collect(bq: bigquery.Client, table: str, features: list[str]) -> pd.DataFrame:
    row = dict(next(iter(bq.query(_stats_sql(table, features)).result())))
    records = {}
    for key, value in row.items():
        feature, stat = key.rsplit(STAT_SEP, 1)
        records.setdefault(feature, {})[stat] = value
    return pd.DataFrame(records).T


def compute_drift(bq: bigquery.Client | None = None) -> pd.DataFrame:
    bq = bq or client()
    cfg = load_config()
    p, f = cfg.bigquery.project, cfg.bigquery.datasets.features
    train_tbl, test_tbl = f"{p}.{f}.dataset_train", f"{p}.{f}.dataset_test"

    features = sorted(set(_feature_list(bq, train_tbl)) & set(_feature_list(bq, test_tbl)))
    log.info("comparing %d features shared by both splits", len(features))

    tr = _collect(bq, train_tbl, features).add_prefix("train_")
    te = _collect(bq, test_tbl, features).add_prefix("test_")
    df = tr.join(te)

    df["shift"] = (df["test_mean"] - df["train_mean"]).abs() / df["train_std"].replace(0, pd.NA)
    df["std_ratio"] = df["test_std"] / df["train_std"].replace(0, pd.NA)
    df["null_delta"] = df["test_null"] - df["train_null"]
    df["family"] = pd.Series(df.index, index=df.index).str.split("_").str[0]

    df = df.reset_index(names="feature")
    return df.sort_values("shift", ascending=False, na_position="last").reset_index(drop=True)


def summarise(df: pd.DataFrame) -> dict:
    """Bucket the effect sizes so the headline is one readable line."""
    s = df["shift"].dropna()
    return {
        "n_features": int(len(df)),
        "negligible_lt_0p1": int((s < 0.1).sum()),
        "small_0p1_0p3": int(((s >= 0.1) & (s < 0.3)).sum()),
        "moderate_0p3_0p5": int(((s >= 0.3) & (s < 0.5)).sum()),
        "large_ge_0p5": int((s >= 0.5).sum()),
        "median_shift": float(s.median()),
        "max_shift": float(s.max()),
        "n_null_delta_gt_0p1": int((df["null_delta"].abs() > 0.1).sum()),
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Train vs test feature drift report")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--publish", action="store_true", help="also write to the BigQuery mart")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    from pathlib import Path

    cfg = load_config()
    df = compute_drift()
    path = Path(cfg.paths.features) / "drift_report.csv"
    df.to_csv(path, index=False)
    log.info("drift report written: %s", path)

    stats = summarise(df)
    log.info("summary: %s", stats)
    log.info("shift by family (mean):\n%s",
             df.groupby("family")["shift"].mean().sort_values(ascending=False).to_string())
    print("\nMost shifted features:")
    print(df.head(args.top)[
        ["feature", "train_mean", "test_mean", "shift", "std_ratio", "null_delta"]
    ].to_string(index=False))

    if args.publish:
        from src.data.mart import _write

        _write(client(), "mart_drift", df, "drift")


if __name__ == "__main__":
    main()

"""Export the small result artefacts the dashboard needs, so it can run anywhere.

The dashboard reads from `C:/mscapital_data`, which exists on one laptop. To publish it,
the numbers it displays have to travel with the repository.

WHAT TRAVELS, AND WHAT DOES NOT

Only DERIVED AGGREGATES: fold scores, backtest curves, SHAP importances, drift statistics,
the investigation results. Together they are a few hundred kilobytes, and none of it is
competition data - it is the output of analysis over that data, which is the same thing
already published in the notebooks and the README.

The competition data itself is never exported. Neither is `dataset_train.parquet`, except
for a deliberately small sample: 5,000 rows - 0.4% of the training set - carrying no labels
beyond the target the notebooks already publish. It is full width rather than the ten
columns the charts plot, because the Predictions page has to assemble a complete
292-feature row and Predictor rejects an incomplete one instead of quietly imputing.

The trained model travels too. A published dashboard has no FastAPI beside it, and a
Predictions page that could only report "cannot reach the API" would be dead on the one
deployment anybody sees.

The equity curve is DOWNSAMPLED. It has 20,954 points, and a line chart cannot show them;
carrying two megabytes to draw a shape that 2,000 points draws identically is waste, not
fidelity.
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_config

log = logging.getLogger(__name__)

REPO_RESULTS = Path(__file__).resolve().parents[2] / "results"

# Copied verbatim - each is small and the dashboard reads it directly.
VERBATIM = [
    "walkforward_summary.json", "walkforward_summary.csv", "holdout_metrics.json",
    "backtest_cost_sensitivity.csv", "backtest_trade_fraction.csv",
    "ablation_subsets.csv", "ablation_marginal.csv", "ablation_meta.json",
    "period_difficulty.csv", "period_difficulty_meta.json",
    "cosine_decomposition.csv", "cosine_decomposition_meta.json",
    "cosine_forecast_corrected.csv",
    "shape_gain.csv", "shape_gain_meta.json", "shape_redundancy.csv", "shape_audit.json",
    "adversarial_auc.csv", "drift_robustness_t020.csv", "drift_robustness_t010.csv",
    "tuning_result.json", "tuning_confirm.json",
    "metric_alignment.csv", "metric_alignment_meta.json",
    "recency.csv", "recency_meta.json",
    "feature_audit.json", "feature_redundancy.csv",
]

# FULL WIDTH, deliberately short. The overview charts only need a few columns, but the
# Predictions page has to build a complete 292-feature row - and Predictor rejects an
# incomplete one rather than quietly imputing, which is the right behaviour and makes a
# narrow sample useless there. 5k rows of every column costs less than the model does.
SAMPLE_COLUMNS = None
SAMPLE_ROWS = 5_000
EQUITY_POINTS = 2_000


def _stratified_index(months: pd.Series, n: int) -> np.ndarray:
    """Row positions spread evenly over every month, rather than the first n rows.

    This was `head(n)`, and because sample_id is chronological that meant the exported
    sample was ENTIRELY month 0. Every distribution on the published dashboard was
    therefore one month of data captioned as though it were the training set, and the
    monthly-volatility chart had a single point under a caption claiming a 2.69x swing.

    Even quotas rather than a proportional draw: the months are near-identical in size
    (17,187 to 17,852 rows), so the two agree to within a few rows, and a fixed quota
    cannot leave a month out.
    """
    per = max(1, n // months.nunique())
    picked = [np.flatnonzero(months.to_numpy() == m)[:per]
              for m in np.sort(months.unique())]
    return np.concatenate(picked)[:n]


def _target_by_month(src: Path, dst: Path) -> int:
    """Per-month target statistics, computed on the FULL table.

    Read from the whole 1.26M rows rather than from the exported sample, because the
    point of the chart is the regime shift ACROSS months and a sample cannot carry it
    faithfully. Two columns of 1.26M rows is a cheap read and 71 rows to carry.
    """
    import pyarrow.parquet as pq

    df = pq.read_table(src, columns=["month", "target"]).to_pandas()
    stats = df.groupby("month")["target"].agg(["std", "mean", "count"]).reset_index()
    stats.to_csv(dst, index=False)
    return len(stats)


def _downsample_equity(src: Path, dst: Path) -> int:
    """Thin the equity curve to a fixed number of points, keeping the endpoints.

    Even spacing rather than a rolling mean: this is a cumulative path, and smoothing it
    would flatten the drawdown the chart exists to show.
    """
    eq = pd.read_csv(src)
    if len(eq) > EQUITY_POINTS:
        idx = np.unique(np.linspace(0, len(eq) - 1, EQUITY_POINTS).round().astype(int))
        eq = eq.iloc[idx].reset_index(drop=True)
    eq.to_csv(dst, index=False)
    return len(eq)


def run(*, out: Path | None = None) -> dict:
    cfg = load_config()
    src_dir = Path(cfg.paths.features)
    models_dir = Path(cfg.paths.data_root) / "models" / "current"
    out = out or REPO_RESULTS
    out.mkdir(parents=True, exist_ok=True)

    copied, skipped = [], []
    for name in VERBATIM:
        p = src_dir / name
        if p.exists():
            shutil.copy2(p, out / name)
            copied.append(name)
        else:
            skipped.append(name)

    # model.txt travels too, so the dashboard can predict without a running API. A
    # published dashboard has no FastAPI beside it, and a Predictions page that can only
    # show "cannot reach the API" is worse than no page at all.
    for name in ("shap_global.csv", "shap_local_examples.csv", "model_meta.json",
                 "model.txt"):
        p = models_dir / name
        if p.exists():
            shutil.copy2(p, out / name)
            copied.append(name)
        else:
            skipped.append(name)

    eq = src_dir / "backtest_equity.csv"
    if eq.exists():
        n = _downsample_equity(eq, out / "backtest_equity.csv")
        copied.append(f"backtest_equity.csv ({n} points, downsampled)")

    # A small slice of the feature table, for the overview charts only.
    ds = src_dir / "dataset_train.parquet"
    if ds.exists():
        import pyarrow.parquet as pq

        months = pq.read_table(ds, columns=["month"]).to_pandas()["month"]
        idx = _stratified_index(months, SAMPLE_ROWS)
        table = pq.read_table(ds, columns=SAMPLE_COLUMNS)
        df = table.take(idx).to_pandas()
        del table
        df = df.astype({c: "float32" for c in df.columns if df[c].dtype == "float64"})
        df.to_parquet(out / "feature_sample.parquet", compression="zstd", index=False)
        copied.append(f"feature_sample.parquet ({len(df):,} rows x {df.shape[1]}, "
                      f"{df['month'].nunique()} months)")

        n = _target_by_month(ds, out / "target_by_month.csv")
        copied.append(f"target_by_month.csv ({n} months, from the full table)")

    total = sum(f.stat().st_size for f in out.glob("*"))
    log.info("exported %d files to %s (%.1f KB)", len(list(out.glob('*'))), out, total / 1024)
    if skipped:
        log.warning("not found, skipped: %s", skipped)
    if total > 24 * 1024 * 1024:
        raise AssertionError(
            f"results/ is {total/1e6:.1f} MB - too large to carry in the repository. "
            "Something big slipped in; check the sample row count and the equity curve."
        )
    return {"copied": copied, "skipped": skipped, "bytes": total}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Export dashboard results into the repo")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    res = run(out=Path(args.out) if args.out else None)
    print(json.dumps({"files": len(res["copied"]), "bytes": res["bytes"]}, indent=2))


if __name__ == "__main__":
    main()

"""`make demo` - run the whole project end to end on synthetic data, in ~3 minutes.

WHY: the real pipeline needs Kaggle credentials, a GCP project, ~20 GB of disk and
hours of upload. Without this, nobody can run the repository, which is a poor property
for a project meant to be read.

WHAT IT ACTUALLY EXERCISES (the genuine code paths, not a mock):
  1. the column-group feather -> Parquet converter, on single-record-batch files
  2. the walk-forward harness, with its embargo and its runtime leakage guard
  3. every model, the cosine metric, and the closed-form ensemble
  4. hold-out measurement and backtesting
  5. artefact saving in the format the API serves

WHAT IT DOES NOT EXERCISE: the BigQuery feature SQL, which needs GCP. That layer is
covered structurally by tests/test_feature_sql.py, and the demo's feature table takes
its COLUMN NAMES from the same SQL generators so the schema cannot drift.

The data is synthetic and the signal in it is planted. No claim in this repository
rests on anything produced here.
"""
from __future__ import annotations

import argparse
import logging
import shutil
import time
from pathlib import Path

from src.config import load_config

log = logging.getLogger(__name__)


def _banner(step: str, text: str) -> None:
    log.info("")
    log.info("=" * 68)
    log.info("  %s  %s", step, text)
    log.info("=" * 68)


def run(samples: int = 4000, keep: bool = False,
        models: list[str] | None = None) -> dict:
    cfg = load_config()
    root = Path(cfg.paths.data_root) / "demo"
    if root.exists() and not keep:
        shutil.rmtree(root)
    raw, features, models_dir = root / "raw", root / "features", root / "models"
    for d in (raw, features, models_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Point the whole stack at the demo directory for this process only.
    cfg["paths"]["raw"] = str(raw)
    cfg["paths"]["parquet"] = str(root / "parquet")
    cfg["paths"]["features"] = str(features)
    cfg["paths"]["data_root"] = str(root)

    t_start = time.perf_counter()
    results: dict = {}

    _banner("1/5", f"generating synthetic data ({samples:,} samples)")
    from src.data import synthetic

    synthetic.write_raw(raw, n_samples=samples)

    _banner("2/5", "column-group ingest (the real converter, on single-batch files)")
    from src.data import ingestion

    for table in ("market", "order", "transaction"):
        for m in ingestion.convert_table("train", table):
            log.info("  %-12s %-3s %s rows -> %d parquet part(s)",
                     table, m["group"], f"{m['rows']:,}", len(m["files"]))
    results["ingest_ok"] = True

    _banner("3/5", "feature table (schema read from the real SQL generators)")
    df = synthetic.make_feature_table(n_samples=samples)
    df.to_parquet(features / "dataset_train.parquet", compression="zstd", index=False)
    log.info("  %s rows x %s columns", f"{len(df):,}", df.shape[1])
    results["n_features"] = df.shape[1] - 3

    _banner("4/5", "walk-forward training (embargo + runtime leakage guard)")
    from src.models.train import build_model_factories, results_frame, run_walk_forward

    summary = run_walk_forward(
        build_model_factories(models or ["zero", "mean", "ridge", "lightgbm"], quick=True),
        df=df, log_mlflow=False,
    )
    table = results_frame(summary)
    log.info("\n%s", table.to_string(index=False))
    results["scores"] = table.set_index("model")["cosine_mean"].to_dict()

    _banner("5/5", "final model, hold-out measurement, servable artefact")
    import numpy as np

    from src.evaluation.metrics import evaluate
    from src.evaluation.temporal_validation import holdout_months
    from src.inference.predictor import Predictor, save_bundle
    from src.models.base import feature_columns
    from src.models.lightgbm_model import LightGBMModel

    lo, hi = holdout_months()
    months, y = df["month"].to_numpy(), df["target"].to_numpy()
    tr = np.flatnonzero(months < lo - 1)
    va = np.flatnonzero(months == lo - 1)
    ho = np.flatnonzero((months >= lo) & (months <= hi))
    assert months[tr].max() < lo and months[va].max() < lo, "hold-out leaked into training"

    model = LightGBMModel(num_boost_round=300, early_stopping_rounds=50)
    model.fit(df.iloc[tr], y[tr], eval_set=(df.iloc[va], y[va]))
    scores = evaluate(y[ho], model.predict(df.iloc[ho]))
    log.info("  hold-out (months %d-%d, %s samples): %s", lo, hi, f"{len(ho):,}",
             {k: round(v, 5) for k, v in scores.items()})
    results["holdout"] = scores

    bundle_dir = models_dir / "current"
    save_bundle(bundle_dir, model=model.booster_, kind="lightgbm",
                features=feature_columns(df), name="lightgbm", version="demo",
                metrics={k: round(v, 6) for k, v in scores.items()})
    served = Predictor.from_dir(bundle_dir)
    row = {c: (0.0 if df.iloc[0][c] != df.iloc[0][c] else float(df.iloc[0][c]))
           for c in feature_columns(df)}
    value = float(served.predict([row])[0])
    log.info("  artefact saved and reloaded; sample prediction %+.6f (%s)",
             value, Predictor.direction(value))
    results["artefact"] = str(bundle_dir)

    log.info("")
    log.info("done in %.0f s", time.perf_counter() - t_start)
    log.info("")
    log.info("  serve it:      MSCAPITAL_MODEL_DIR=%s \\", bundle_dir)
    log.info("                 python -m uvicorn api.main:app --port 8000")
    log.info("  dashboard:     MSCAPITAL_DATA_ROOT=%s \\", root)
    log.info("                 python -m streamlit run streamlit_app/app.py")
    log.info("")
    log.info("  Reminder: this data is synthetic and its signal is planted.")
    return results


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="End-to-end demo on synthetic data")
    ap.add_argument("--samples", type=int, default=4000)
    ap.add_argument("--keep", action="store_true", help="do not wipe the demo directory")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run(samples=args.samples, keep=args.keep)


if __name__ == "__main__":
    main()

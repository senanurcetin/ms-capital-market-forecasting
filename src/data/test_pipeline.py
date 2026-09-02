"""End-to-end test-set pipeline: feather -> parquet -> BigQuery -> features -> submission.

It reuses the SAME SQL generators as train, which makes a feature-definition drift
between train and test impossible. The only difference: test has NO month and NO
target (the competition does not provide them), so staging is partitioned by
sample_id buckets instead.

Every step is idempotent, so an interrupted run can simply be restarted.
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from src.config import load_config
from src.data import ingestion, staging
from src.data.bq_loader import client, ensure_datasets, load_table
from src.features import assemble

log = logging.getLogger(__name__)

TABLES = ("market", "order", "transaction")


def step_verify_source() -> None:
    for table in TABLES:
        ok, msg = ingestion.verify_source("test", table)
        if not ok:
            raise SystemExit(f"test/{table}: {msg}")
        log.info("[source] test/%s %s", table, msg)


def step_convert() -> None:
    for table in TABLES:
        t0 = time.perf_counter()
        for m in ingestion.convert_table("test", table):
            log.info("[parquet] test/%s/%s %s rows, %d files, %.2f GB (%.0fs)",
                     table, m["group"], f"{m['rows']:,}", len(m["files"]),
                     m["total_bytes"] / 1e9, time.perf_counter() - t0)


def step_upload() -> None:
    bq = client()
    ensure_datasets(bq)
    for table in TABLES:
        for r in load_table("test", table):
            log.info("[bq] %s %s rows (%.0fs, %.1f MB)",
                     r["table_id"], f"{r['rows']:,}", r["elapsed_s"], r["uploaded_mb"])


def step_staging() -> None:
    bq = client()
    staging.assert_group_alignment("test", bq)
    for name, table in (
        ("staging_market.sql", "market"),
        ("staging_order.sql", "order"),
        ("staging_transaction.sql", "transaction"),
    ):
        r = staging.run_sql_file(name, "test", bq=bq)
        log.info("[staging] %s scanned %.1f GB", table, r["bytes_processed"] / 1e9)
    for row in staging.verify_staging("test", bq):
        log.info("[verify] %s", row)


def step_features() -> None:
    bq = client()
    for name, info in assemble.build_blocks("test", bq=bq).items():
        log.info("[feature] %s %s rows x %s columns (%.1f GB scanned)",
                 name, f"{info['rows']:,}", info["columns"], info["gb_scanned"])
    r = assemble.assemble("test", bq=bq)
    log.info("[dataset] %s rows x %s columns", f"{r['rows']:,}", r["columns"])
    assemble.download("test", bq=bq)


def step_submission(model_version: str = "v1") -> Path:
    """Produce submission.csv using the saved model artefact."""
    import numpy as np
    import pandas as pd

    from src.inference.predictor import load_bundle

    cfg = load_config()
    bundle = load_bundle(Path(cfg.paths.data_root) / "models" / "current")
    df = pd.read_parquet(Path(cfg.paths.features) / "dataset_test.parquet")
    if len(df) != cfg.samples["test"]:
        raise SystemExit(f"test rows {len(df):,} != {cfg.samples['test']:,}")

    missing = [c for c in bundle.features if c not in df.columns]
    if missing:
        raise SystemExit(f"{len(missing)} feature(s) missing in test: {missing[:5]}")

    X = df[bundle.features].astype("float64")
    model = bundle.model
    if model.__class__.__module__.startswith("xgboost"):
        import xgboost as xgb

        pred = model.predict(xgb.DMatrix(X))
    else:
        pred = model.predict(X)

    out = pd.DataFrame({"sample_id": df["sample_id"].astype(int),
                        "prediction": np.asarray(pred, dtype=np.float64)})
    out = out.sort_values("sample_id").reset_index(drop=True)

    # The sample_id set must match the Kaggle submission template exactly.
    template = pd.read_csv(Path(cfg.paths.raw) / "submission.csv")
    if set(out["sample_id"]) != set(template["sample_id"]):
        raise SystemExit("sample_id set does not match the submission template")

    path = Path(cfg.paths.features) / "submission.csv"
    out.to_csv(path, index=False)
    log.info("[submission] %s | %s rows | pred std %.6f | mean %.2e",
             path, f"{len(out):,}", out["prediction"].std(), out["prediction"].mean())
    return path


STEPS = {
    "verify": step_verify_source,
    "convert": step_convert,
    "upload": step_upload,
    "staging": step_staging,
    "features": step_features,
    "submission": step_submission,
}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Test-set pipeline")
    ap.add_argument("--steps", default=",".join(STEPS),
                    help=f"comma-separated: {','.join(STEPS)}")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    for name in args.steps.split(","):
        if name not in STEPS:
            raise SystemExit(f"unknown step: {name}")
        log.info("=== STEP: %s ===", name)
        t0 = time.perf_counter()
        STEPS[name]()
        log.info("=== %s done (%.0fs) ===", name, time.perf_counter() - t0)


if __name__ == "__main__":
    main()

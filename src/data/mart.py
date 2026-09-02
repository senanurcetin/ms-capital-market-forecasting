"""Sonuc katmani: model metrikleri, backtest ve SHAP ciktilarini BigQuery mart'a yazar.

Neden ayri bir katman: egitim lokalde kompakt veriyle kosuyor, ama sonuclarin
tek bir sorgulanabilir yerde durmasi gerekiyor - Streamlit, dbt ve raporlama
buradan okur. Tablolar kucuk (yuzlerce satir), maliyeti ihmal edilebilir.

Her calistirmada tablolar WRITE_TRUNCATE ile yenilenir ve run_id ile
damgalanir; boylece "hangi sonuc hangi kosudan geldi" kaybolmaz.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
from pathlib import Path

import pandas as pd
from google.cloud import bigquery

from src.config import load_config
from src.data.bq_loader import client

log = logging.getLogger(__name__)

TABLES = {
    "mart_model_metrics": "Walk-forward: model x fold metrikleri",
    "mart_holdout_metrics": "Hold-out (ay 65-70) nihai olcum",
    "mart_backtest": "Backtest: islem maliyeti ve esik duyarliligi",
    "mart_feature_importance": "SHAP global feature onemi",
    "mart_run_metadata": "Kosu kunyesi",
}


def _write(bq: bigquery.Client, name: str, df: pd.DataFrame, run_id: str) -> int:
    if df.empty:
        log.warning("[%s] bos, atlandi", name)
        return 0
    cfg = load_config()
    table_id = f"{cfg.bigquery.project}.{cfg.bigquery.datasets.mart}.{name}"
    df = df.copy()
    df["run_id"] = run_id
    job = bq.load_table_from_dataframe(
        df, table_id,
        job_config=bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE"),
    )
    job.result()
    n = bq.get_table(table_id).num_rows
    log.info("[%s] %s satir yazildi", name, f"{n:,}")
    return n


def _walkforward(features_dir: Path) -> pd.DataFrame:
    rows = []
    for fname in ("walkforward_summary.json", "smoke_summary.json"):
        p = features_dir / fname
        if not p.exists():
            continue
        summary = json.loads(p.read_text(encoding="utf-8"))
        for model, blk in summary.items():
            stab = blk.get("stability", {})
            for r in blk.get("per_fold", []):
                rows.append({
                    "source": fname,
                    "model": model,
                    "fold": r.get("fold"),
                    "cosine": r.get("cosine", r.get("ensemble_score")),
                    "mae": r.get("mae"),
                    "rmse": r.get("rmse"),
                    "pearson": r.get("pearson"),
                    "directional_accuracy": r.get("directional_accuracy"),
                    "train_rows": r.get("train_rows"),
                    "val_rows": r.get("val_rows"),
                    "train_seconds": r.get("train_seconds"),
                    "best_iteration": r.get("best_iteration"),
                    "cosine_mean_all_folds": stab.get("mean"),
                    "cosine_std_all_folds": stab.get("std"),
                })
        break
    return pd.DataFrame(rows)


def _holdout(features_dir: Path) -> pd.DataFrame:
    p = features_dir / "holdout_metrics.json"
    if not p.exists():
        return pd.DataFrame()
    d = json.loads(p.read_text(encoding="utf-8"))
    row = {"model": d.get("model"), **d.get("scores", {})}
    bt = d.get("backtest", {})
    for k in ("n_trades", "turnover", "total_return", "mean_return", "sharpe",
              "max_drawdown", "win_rate", "cost_bps", "threshold"):
        if k in bt:
            row[f"bt_{k}"] = bt[k]
    return pd.DataFrame([row])


def _backtest(features_dir: Path) -> pd.DataFrame:
    frames = []
    for fname, kind in (("backtest_cost_sensitivity.csv", "cost_sensitivity"),
                        ("backtest_trade_fraction.csv", "trade_fraction")):
        p = features_dir / fname
        if p.exists():
            df = pd.read_csv(p)
            df.insert(0, "analysis", kind)
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _importance(models_dir: Path) -> pd.DataFrame:
    p = models_dir / "current" / "shap_global.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df


def publish(run_id: str | None = None) -> dict[str, int]:
    cfg = load_config()
    bq = client()
    features_dir = Path(cfg.paths.features)
    models_dir = Path(cfg.paths.data_root) / "models"
    run_id = run_id or dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    meta_path = models_dir / "current" / "model_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

    written = {
        "mart_model_metrics": _write(bq, "mart_model_metrics", _walkforward(features_dir), run_id),
        "mart_holdout_metrics": _write(bq, "mart_holdout_metrics", _holdout(features_dir), run_id),
        "mart_backtest": _write(bq, "mart_backtest", _backtest(features_dir), run_id),
        "mart_feature_importance": _write(
            bq, "mart_feature_importance", _importance(models_dir), run_id),
    }
    written["mart_run_metadata"] = _write(bq, "mart_run_metadata", pd.DataFrame([{
        "published_at": dt.datetime.now(dt.timezone.utc),
        "model_name": meta.get("name"),
        "model_version": meta.get("version"),
        "n_features": len(meta.get("features", [])),
        "trained_at": meta.get("trained_at"),
        "holdout_months": meta.get("metrics", {}).get("holdout_months"),
        "holdout_cosine": meta.get("metrics", {}).get("cosine"),
        "train_samples": cfg.samples["train"],
        "test_samples": cfg.samples["test"],
        "market_window_seconds": cfg.window.seconds["market"],
        "order_window_seconds": cfg.window.seconds["order"],
    }]), run_id)
    log.info("run_id=%s | yazilan: %s", run_id, written)
    return written


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Sonuclari BigQuery mart'a yayinla")
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    publish(args.run_id)


if __name__ == "__main__":
    main()

"""Walk-forward training harness with MLflow tracking.

Single entry point: run_walk_forward(). Every model sees the same fold structure,
which is what makes the comparison fair.

LEAKAGE GUARD (re-checked at runtime, never merely trusted):
  * a fold's train months are STRICTLY before its validation months
  * the embargo months in between are used by neither side
  * hold-out months (65-70) appear in no fold
  assert_fold_integrity() re-verifies all of this against the data on every fold.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from src.config import load_config
from src.evaluation.metrics import evaluate
from src.evaluation.temporal_validation import (
    Fold, holdout_months, iter_folds, stability,
)
from src.models.base import feature_columns
from src.models.ensemble import evaluate_ensemble_gain

log = logging.getLogger(__name__)


def load_dataset(
    split: str = "train",
    columns: list[str] | None = None,
    *,
    float32: bool = True,
) -> pd.DataFrame:
    """Load the compact feature table.

    float32 by default, which HALVES peak memory: 1.26M rows x 294 features is
    2.96 GB as float64 but 1.48 GB as float32, and pandas needs roughly twice that
    transiently while reading. On a 16 GB machine the float64 path can fail outright
    if anything else is running - it did during development.

    No precision is lost that matters: the source columns are float32 in the original
    feather files, and LightGBM bins features into uint8 internally anyway. Metrics
    still promote to float64 (see evaluation/metrics.py), so scoring is unaffected.
    """
    cfg = load_config()
    path = Path(cfg.paths.features) / f"dataset_{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found - run src/features/assemble.py:download('{split}') first"
        )
    if not float32:
        return pd.read_parquet(path, columns=columns)

    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=columns)
    import pyarrow as pa

    cast = []
    for field in table.schema:
        if field.name in ("sample_id", "month"):
            cast.append(field)
        elif pa.types.is_floating(field.type):
            cast.append(pa.field(field.name, pa.float32()))
        else:
            cast.append(field)
    table = table.cast(pa.schema(cast))
    df = table.to_pandas(split_blocks=True, self_destruct=True)
    del table
    return df


def assert_fold_integrity(months: np.ndarray, fold: Fold,
                          tr: np.ndarray, va: np.ndarray) -> None:
    """Verify the split against the DATA, not against the config."""
    ho_lo, ho_hi = holdout_months()
    m_tr, m_va = months[tr], months[va]
    if m_tr.max() >= m_va.min():
        raise AssertionError(f"{fold.describe()}: train is not strictly before validation")
    gap_lo, gap_hi = fold.embargo_months
    for m in range(gap_lo, gap_hi + 1):
        if (m_tr == m).any() or (m_va == m).any():
            raise AssertionError(f"{fold.describe()}: embargo month {m} was used")
    if (m_tr >= ho_lo).any() or (m_va >= ho_lo).any():
        raise AssertionError(f"{fold.describe()}: hold-out ({ho_lo}-{ho_hi}) leaked in")
    if set(tr) & set(va):
        raise AssertionError(f"{fold.describe()}: train and validation indices overlap")


def run_walk_forward(
    model_factories: dict[str, Callable[[], object]],
    df: pd.DataFrame | None = None,
    *,
    experiment: str | None = None,
    log_mlflow: bool = True,
) -> dict:
    cfg = load_config()
    df = load_dataset("train") if df is None else df
    months = df["month"].to_numpy()
    y_all = df["target"].to_numpy()
    feats = feature_columns(df)
    log.info("dataset: %s rows x %s features", f"{len(df):,}", len(feats))

    mlflow = None
    if log_mlflow:
        import mlflow as _mlflow

        mlflow = _mlflow
        # SQLite backend, not the bare filesystem store: MLflow 3.x puts './mlruns'
        # file stores in maintenance mode and refuses to open them. SQLite is also
        # what docker-compose serves, so local and containerised tracking agree.
        root = Path(cfg.paths.mlruns)
        root.mkdir(parents=True, exist_ok=True)
        (root / "artifacts").mkdir(exist_ok=True)
        mlflow.set_tracking_uri(f"sqlite:///{(root / 'mlflow.db').as_posix()}")
        name = experiment or cfg.mlflow.experiment
        if mlflow.get_experiment_by_name(name) is None:
            mlflow.create_experiment(name, artifact_location=(root / "artifacts").as_uri())
        mlflow.set_experiment(name)

    results: dict[str, list[dict]] = {name: [] for name in model_factories}
    fold_preds: list[dict[str, np.ndarray]] = []
    fold_truth: list[np.ndarray] = []

    for fold, tr, va in iter_folds(months):
        assert_fold_integrity(months, fold, tr, va)
        log.info("%s | train %s / val %s", fold.describe(), f"{len(tr):,}", f"{len(va):,}")
        Xtr, ytr = df.iloc[tr], y_all[tr]
        Xva, yva = df.iloc[va], y_all[va]
        preds_this_fold: dict[str, np.ndarray] = {}

        for name, factory in model_factories.items():
            t0 = time.perf_counter()
            model = factory()
            try:
                model.fit(Xtr, ytr, eval_set=(Xva, yva))
            except TypeError:
                model.fit(Xtr, ytr)
            pred = model.predict(Xva)
            elapsed = time.perf_counter() - t0
            scores = evaluate(yva, pred)
            row = {
                "fold": fold.index, "model": name, "train_rows": len(tr),
                "val_rows": len(va), "train_seconds": round(elapsed, 1),
                "best_iteration": getattr(model, "best_iteration_", None), **scores,
            }
            results[name].append(row)
            preds_this_fold[name] = pred
            log.info("  %-10s cosine=%+.5f rmse=%.6f dir_acc=%.4f (%.0fs)",
                     name, scores["cosine"], scores["rmse"],
                     scores["directional_accuracy"], elapsed)

            if mlflow is not None:
                with mlflow.start_run(run_name=f"{name}_fold{fold.index}"):
                    mlflow.log_params({
                        "model": name, "fold": fold.index,
                        "train_months": f"{fold.train_months[0]}-{fold.train_months[1]}",
                        "val_months": f"{fold.val_months[0]}-{fold.val_months[1]}",
                        "embargo_months": f"{fold.embargo_months[0]}-{fold.embargo_months[1]}",
                        "n_features": len(feats),
                        **{f"hp_{k}": v for k, v in getattr(model, "params", {}).items()},
                    })
                    mlflow.log_metrics({**scores, "train_seconds": elapsed})

        fold_preds.append(preds_this_fold)
        fold_truth.append(yva)

    summary = {}
    for name, rows in results.items():
        cos = [r["cosine"] for r in rows]
        summary[name] = {"per_fold": rows, "stability": stability(cos)}
        s = summary[name]["stability"]
        log.info("%-10s cosine mean=%+.5f std=%.5f min=%+.5f (worst fold %d)",
                 name, s["mean"], s["std"], s["min"], s["worst_fold"])

    # Ensemble: does it genuinely beat the best single model, fold by fold?
    ens_rows = []
    for i, (preds, truth) in enumerate(zip(fold_preds, fold_truth), start=1):
        if len(preds) > 1:
            gain = evaluate_ensemble_gain(preds, truth)
            gain["fold"] = i
            ens_rows.append(gain)
            log.info("fold %d ensemble=%+.5f vs best single (%s)=%+.5f  gain=%+.5f  %s",
                     i, gain["ensemble_score"], gain["best_single"],
                     gain["best_single_score"], gain["gain"],
                     "BEATS IT" if gain["beats_best_single"] else "does not beat it")
    if ens_rows:
        summary["ensemble"] = {
            "per_fold": ens_rows,
            "stability": stability([r["ensemble_score"] for r in ens_rows]),
            "beats_best_single_in_folds": sum(r["beats_best_single"] for r in ens_rows),
            "n_folds": len(ens_rows),
        }
    return summary


def results_frame(summary: dict) -> pd.DataFrame:
    rows = []
    for name, blk in summary.items():
        s = blk["stability"]
        rows.append({"model": name, "cosine_mean": s["mean"], "cosine_std": s["std"],
                     "cosine_min": s["min"], "cosine_max": s["max"],
                     "worst_fold": s["worst_fold"]})
    return pd.DataFrame(rows).sort_values("cosine_mean", ascending=False)


# --------------------------------------------------------------------------
# CLI: python -m src.models.train [--quick] [--models ridge,lightgbm,xgboost]
# --------------------------------------------------------------------------
def build_model_factories(names: list[str], quick: bool) -> dict:
    from src.models.baseline import MeanPredictor, RidgeModel, ZeroPredictor
    from src.models.lightgbm_model import LightGBMModel
    from src.models.xgboost_model import XGBoostModel

    rounds = 400 if quick else 3000
    stop = 50 if quick else 150
    available = {
        "zero": ZeroPredictor,
        "mean": MeanPredictor,
        "ridge": lambda: RidgeModel(alpha=10.0),
        "lightgbm": lambda: LightGBMModel(num_boost_round=rounds, early_stopping_rounds=stop),
        "xgboost": lambda: XGBoostModel(num_boost_round=rounds, early_stopping_rounds=stop),
    }
    unknown = set(names) - set(available)
    if unknown:
        raise SystemExit(f"unknown model(s): {sorted(unknown)}; choose from {sorted(available)}")
    return {n: available[n] for n in names}


def main(argv: list[str] | None = None) -> None:
    import argparse
    import json
    import warnings

    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(description="MSCapital walk-forward training")
    ap.add_argument("--models", default="zero,mean,ridge,lightgbm,xgboost")
    ap.add_argument("--quick", action="store_true", help="fewer trees, fast sanity run")
    ap.add_argument("--folds", type=int, default=0, help="0 = all folds, N = last N folds")
    ap.add_argument("--sample-frac", type=float, default=0.0, help="0 = full data")
    ap.add_argument("--no-mlflow", action="store_true")
    ap.add_argument("--out", default=None, help="summary JSON path")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg = load_config()
    df = load_dataset("train")
    if args.sample_frac:
        # Sample WITHIN each month so the temporal structure is preserved.
        df = df.groupby("month", group_keys=False).sample(
            frac=args.sample_frac, random_state=42
        )
        log.info("sampled: %s rows (frac=%.3f)", f"{len(df):,}", args.sample_frac)

    if args.folds:
        from src.evaluation import temporal_validation as tv

        original = tv.build_folds
        tv.build_folds = lambda: original()[-args.folds :]
        globals()["iter_folds"] = tv.iter_folds

    summary = run_walk_forward(
        build_model_factories(args.models.split(","), args.quick),
        df=df, log_mlflow=not args.no_mlflow,
    )
    table = results_frame(summary)
    print("\n" + table.to_string(index=False))

    out = Path(args.out) if args.out else Path(cfg.paths.features) / "walkforward_summary.json"
    serialisable = {
        k: {"stability": v["stability"],
            "per_fold": [{kk: vv for kk, vv in r.items() if kk != "weights"}
                         for r in v["per_fold"]]}
        for k, v in summary.items()
    }
    out.write_text(json.dumps(serialisable, indent=2, default=str), encoding="utf-8")
    table.to_csv(out.with_suffix(".csv"), index=False)
    log.info("summary written: %s", out)


if __name__ == "__main__":
    main()

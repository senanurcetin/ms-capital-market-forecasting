"""Walk-forward egitim harness'i + MLflow takibi.

Tek giris noktasi: run_walk_forward(). Her model ayni fold yapisini kullanir,
boylece karsilastirma adil olur.

LEAKAGE KORUMASI (calisma aninda dogrulanir, sessizce guvenilmez):
  * fold'un train aylari val aylarindan KESINLIKLE once
  * aralarinda embargo ayi var (ortusen 60sn pencerelerini keser)
  * hold-out aylari (65-70) hicbir fold'da gorunmez
  assert_fold_integrity() bunlari her fold'da yeniden kontrol eder.
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


def load_dataset(split: str = "train", columns: list[str] | None = None) -> pd.DataFrame:
    cfg = load_config()
    path = Path(cfg.paths.features) / f"dataset_{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} yok - once src/features/assemble.py:download('{split}') calistirin"
        )
    return pd.read_parquet(path, columns=columns)


def assert_fold_integrity(months: np.ndarray, fold: Fold,
                          tr: np.ndarray, va: np.ndarray) -> None:
    """Split'in dogrulugunu VERI uzerinde dogrular (config'e guvenmez)."""
    ho_lo, ho_hi = holdout_months()
    m_tr, m_va = months[tr], months[va]
    if m_tr.max() >= m_va.min():
        raise AssertionError(f"{fold.describe()}: train, val'den once degil")
    gap_lo, gap_hi = fold.embargo_months
    for m in range(gap_lo, gap_hi + 1):
        if (m_tr == m).any() or (m_va == m).any():
            raise AssertionError(f"{fold.describe()}: embargo ayi {m} kullanilmis")
    if (m_tr >= ho_lo).any() or (m_va >= ho_lo).any():
        raise AssertionError(f"{fold.describe()}: hold-out ({ho_lo}-{ho_hi}) sizmis")
    if set(tr) & set(va):
        raise AssertionError(f"{fold.describe()}: train/val indeksleri kesisiyor")


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
    log.info("dataset: %s satir x %s feature", f"{len(df):,}", len(feats))

    mlflow = None
    if log_mlflow:
        import mlflow as _mlflow

        mlflow = _mlflow
        mlflow.set_tracking_uri(Path(cfg.paths.mlruns).as_uri())
        mlflow.set_experiment(experiment or cfg.mlflow.experiment)

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
        log.info("%-10s cosine ort=%+.5f std=%.5f min=%+.5f (en kotu fold %d)",
                 name, s["mean"], s["std"], s["min"], s["worst_fold"])

    # Ensemble: fold bazinda tek modeli gercekten geciyor mu?
    ens_rows = []
    for i, (preds, truth) in enumerate(zip(fold_preds, fold_truth), start=1):
        if len(preds) > 1:
            gain = evaluate_ensemble_gain(preds, truth)
            gain["fold"] = i
            ens_rows.append(gain)
            log.info("fold %d ensemble=%+.5f vs en iyi tek (%s)=%+.5f  kazanc=%+.5f  %s",
                     i, gain["ensemble_score"], gain["best_single"],
                     gain["best_single_score"], gain["gain"],
                     "GECIYOR" if gain["beats_best_single"] else "gecmiyor")
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
        raise SystemExit(f"bilinmeyen model: {sorted(unknown)}; secenekler {sorted(available)}")
    return {n: available[n] for n in names}


def main(argv: list[str] | None = None) -> None:
    import argparse
    import json
    import warnings

    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(description="MSCapital walk-forward egitimi")
    ap.add_argument("--models", default="zero,mean,ridge,lightgbm,xgboost")
    ap.add_argument("--quick", action="store_true", help="az agac, hizli dogrulama")
    ap.add_argument("--folds", type=int, default=0, help="0 = tum fold'lar, N = son N fold")
    ap.add_argument("--sample-frac", type=float, default=0.0, help="0 = tam veri")
    ap.add_argument("--no-mlflow", action="store_true")
    ap.add_argument("--out", default=None, help="ozet JSON yolu")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg = load_config()
    df = load_dataset("train")
    if args.sample_frac:
        # Ay yapisini KORUYARAK ornekle - temporal split bozulmamali
        df = df.groupby("month", group_keys=False).sample(
            frac=args.sample_frac, random_state=42
        )
        log.info("ornekleme: %s satir (frac=%.3f)", f"{len(df):,}", args.sample_frac)

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
    log.info("ozet yazildi: %s", out)


if __name__ == "__main__":
    main()

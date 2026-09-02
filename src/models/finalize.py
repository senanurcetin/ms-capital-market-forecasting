"""Train the final model, measure it once on the HOLD-OUT, and write the serving artefact.

The order matters:
  1. Train on months 0-63, early-stop on month 64 (WITHOUT touching the hold-out).
  2. Measure ONCE on months 65-70 (the hold-out) - that is the reported number.
  3. Save the artefact: model + feature order + metrics.

The hold-out is never used for hyperparameter selection; it is only the final read.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from src.config import load_config
from src.evaluation.backtesting import backtest, cost_sensitivity, sweep_trade_fraction
from src.evaluation.metrics import evaluate
from src.evaluation.temporal_validation import holdout_months
from src.inference.predictor import save_bundle
from src.models.base import feature_columns
from src.models.train import load_dataset

log = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Final model + hold-out measurement")
    ap.add_argument("--model", default="lightgbm", choices=["lightgbm", "xgboost", "ridge"])
    ap.add_argument("--rounds", type=int, default=2000)
    ap.add_argument("--early-stopping", type=int, default=100)
    ap.add_argument("--sample-frac", type=float, default=0.0)
    ap.add_argument("--version", default="v1")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg = load_config()
    df = load_dataset("train")
    if args.sample_frac:
        df = df.groupby("month", group_keys=False).sample(
            frac=args.sample_frac, random_state=42
        )

    ho_lo, ho_hi = holdout_months()
    months = df["month"].to_numpy()
    y = df["target"].to_numpy()

    # Early stopping uses the last month BEFORE the hold-out.
    valid_month = ho_lo - 1
    tr = np.flatnonzero(months < valid_month)
    va = np.flatnonzero(months == valid_month)
    ho = np.flatnonzero((months >= ho_lo) & (months <= ho_hi))
    log.info(
        "train months 0-%d (%s) | early stop month %d (%s) | hold-out months %d-%d (%s)",
        valid_month - 1, f"{len(tr):,}", valid_month, f"{len(va):,}",
        ho_lo, ho_hi, f"{len(ho):,}",
    )
    assert months[tr].max() < ho_lo and months[va].max() < ho_lo, "hold-out leaked in"

    if args.model == "lightgbm":
        from src.models.lightgbm_model import LightGBMModel

        model = LightGBMModel(num_boost_round=args.rounds,
                              early_stopping_rounds=args.early_stopping)
        kind = "lightgbm"
    elif args.model == "xgboost":
        from src.models.xgboost_model import XGBoostModel

        model = XGBoostModel(num_boost_round=args.rounds,
                             early_stopping_rounds=args.early_stopping)
        kind = "xgboost"
    else:
        from src.models.baseline import RidgeModel

        model = RidgeModel(alpha=10.0)
        kind = "sklearn"

    model.fit(df.iloc[tr], y[tr], eval_set=(df.iloc[va], y[va]))

    pred_ho = model.predict(df.iloc[ho])
    scores = evaluate(y[ho], pred_ho)
    log.info("HOLD-OUT (months %d-%d): %s", ho_lo, ho_hi,
             {k: round(v, 5) for k, v in scores.items()})

    # Backtest - HOLD-OUT ONLY.
    out_dir = Path(cfg.paths.features)
    out_dir.mkdir(parents=True, exist_ok=True)
    cost_sensitivity(pred_ho, y[ho]).to_csv(
        out_dir / "backtest_cost_sensitivity.csv", index=False)
    sweep_trade_fraction(pred_ho, y[ho]).to_csv(
        out_dir / "backtest_trade_fraction.csv", index=False)
    bt = backtest(pred_ho, y[ho], trade_fraction=0.2, cost_bps=1.0)
    equity = bt.pop("equity_curve")
    import pandas as pd

    pd.DataFrame({"equity": equity}).to_csv(out_dir / "backtest_equity.csv", index=False)
    log.info("backtest (20%% traded, 1 bps): %s",
             {k: (round(v, 6) if isinstance(v, float) else v) for k, v in bt.items()})

    inner = getattr(model, "booster_", None) or getattr(model, "model", model)
    model_dir = Path(cfg.paths.data_root) / "models" / "current"
    save_bundle(
        model_dir, model=inner, kind=kind,
        features=feature_columns(df), name=args.model, version=args.version,
        metrics={**{k: round(v, 6) for k, v in scores.items()},
                 "holdout_months": f"{ho_lo}-{ho_hi}",
                 "backtest_sharpe": round(bt["sharpe"], 4),
                 "backtest_total_return": round(bt["total_return"], 6)},
    )
    (out_dir / "holdout_metrics.json").write_text(
        json.dumps({"model": args.model, "scores": scores, "backtest": bt}, indent=2,
                   default=str), encoding="utf-8")
    log.info("artefact written: %s", model_dir)


if __name__ == "__main__":
    main()

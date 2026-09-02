"""Nihai modeli egitir, HOLD-OUT'ta bir kez olcer ve servis artefaktini yazar.

Sira onemli:
  1. Ay 0-63 ile egit, ay 64 ile erken durdurma (hold-out'a DOKUNMADAN).
  2. Ay 65-70 (hold-out) uzerinde BIR KEZ olc - bu sayi raporlanan sayidir.
  3. Artefakti kaydet: model + feature sirasi + metrikler.

Hold-out hiperparametre secimi icin KULLANILMAZ; sadece son olcumdur.
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
    ap = argparse.ArgumentParser(description="Nihai model + hold-out olcumu")
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

    # Erken durdurma icin hold-out'tan ONCEKI son ay kullanilir.
    valid_month = ho_lo - 1
    tr = np.flatnonzero(months < valid_month)
    va = np.flatnonzero(months == valid_month)
    ho = np.flatnonzero((months >= ho_lo) & (months <= ho_hi))
    log.info(
        "train ay 0-%d (%s) | erken durdurma ay %d (%s) | hold-out ay %d-%d (%s)",
        valid_month - 1, f"{len(tr):,}", valid_month, f"{len(va):,}",
        ho_lo, ho_hi, f"{len(ho):,}",
    )
    assert months[tr].max() < ho_lo and months[va].max() < ho_lo, "hold-out sizmis"

    if args.model == "lightgbm":
        from src.models.lightgbm_model import LightGBMModel

        model = LightGBMModel(num_boost_round=args.rounds,
                              early_stopping_rounds=args.early_stopping)
        kind, inner = "lightgbm", None
    elif args.model == "xgboost":
        from src.models.xgboost_model import XGBoostModel

        model = XGBoostModel(num_boost_round=args.rounds,
                             early_stopping_rounds=args.early_stopping)
        kind, inner = "xgboost", None
    else:
        from src.models.baseline import RidgeModel

        model = RidgeModel(alpha=10.0)
        kind, inner = "sklearn", None

    model.fit(df.iloc[tr], y[tr], eval_set=(df.iloc[va], y[va]))

    pred_ho = model.predict(df.iloc[ho])
    scores = evaluate(y[ho], pred_ho)
    log.info("HOLD-OUT (ay %d-%d): %s", ho_lo, ho_hi,
             {k: round(v, 5) for k, v in scores.items()})

    # Backtest - YALNIZ hold-out'ta
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
    log.info("backtest (%%20 islem, 1 bps): %s",
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
    log.info("artefakt yazildi: %s", model_dir)


if __name__ == "__main__":
    main()

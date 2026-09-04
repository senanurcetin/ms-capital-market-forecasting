"""Build the model that should actually be submitted.

The artefact behind the 0.128 leaderboard score had two handicaps, both visible in its own
metadata rather than inferred:

  1. It was a single LightGBM. This project's own walk-forward CV puts the ensemble ahead
     in 5 folds out of 5, by +0.0022.
  2. It was trained on months 0-63 only. `finalize.py` keeps 65-70 out so the hold-out can
     be read once - correct for MEASURING, wrong for SHIPPING. A hold-out has done its job
     the moment it is read; carrying it through to the deployed artefact discards 7 of 71
     months, and the most recent ones, which sit closest to the test period.

This module fixes both. It deliberately does NOT report a hold-out score, because there is
no longer an untouched period to report one on - and quoting the number from finalize.py
next to a differently-trained model would be a category error. The honest estimate for
this artefact is the walk-forward CV mean; the leaderboard is the test.

THE SPLIT

    months 0-67    train
    month  68      early stopping for each base model
    months 69-70   fit the ensemble weights

Three months are spent instead of seven, and no month does two jobs: fitting stopping
rounds and blend weights on the same rows would let the weights compensate for a stopping
point chosen on those rows.

The blend window is two months rather than one for a reason found the hard way. Fitted on
month 70 alone (17.6k rows), NNLS handed xgboost a weight of 0.71 - while the walk-forward
CV finds lightgbm and xgboost statistically indistinguishable. The three models are highly
correlated, so their DIFFERENCES are almost all noise, and the weight split swings on very
little. Two months roughly halves that variance; equal weights are reported alongside as a
sanity check, since a blend whose weights matter a lot is a blend fitted on too little.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from src.config import load_config
from src.evaluation.metrics import cosine_similarity
from src.inference.predictor import save_bundle
from src.models.base import feature_columns
from src.models.ensemble import CosineOptimalEnsemble
from src.models.train import load_dataset

log = logging.getLogger(__name__)

TRAIN_END = 67
STOP_MONTH = 68
BLEND_MONTHS = (69, 70)


def build(*, rounds: int = 2000, early_stopping: int = 100, version: str = "v4") -> dict:
    from src.models.baseline import RidgeModel
    from src.models.lightgbm_model import LightGBMModel
    from src.models.xgboost_model import XGBoostModel

    cfg = load_config()
    df = load_dataset("train")
    months, y = df["month"].to_numpy(), df["target"].to_numpy()

    tr = np.flatnonzero(months <= TRAIN_END)
    st = np.flatnonzero(months == STOP_MONTH)
    bl = np.flatnonzero((months >= BLEND_MONTHS[0]) & (months <= BLEND_MONTHS[1]))
    assert len(tr) and len(st) and len(bl), "one of the splits is empty"
    assert months[tr].max() < STOP_MONTH < BLEND_MONTHS[0], "splits must not overlap"
    log.info("train 0-%d (%s) | early stop month %d (%s) | blend months %d-%d (%s)",
             TRAIN_END, f"{len(tr):,}", STOP_MONTH, f"{len(st):,}",
             *BLEND_MONTHS, f"{len(bl):,}")
    log.info("that is %d of 71 months in training, against 64 for the submitted artefact",
             TRAIN_END + 1)

    models = {
        "lightgbm": LightGBMModel(num_boost_round=rounds,
                                  early_stopping_rounds=early_stopping),
        "xgboost": XGBoostModel(num_boost_round=rounds,
                                early_stopping_rounds=early_stopping),
        "ridge": RidgeModel(alpha=10.0),
    }

    blend_preds, singles = [], {}
    for name, model in models.items():
        model.fit(df.iloc[tr], y[tr], eval_set=(df.iloc[st], y[st]))
        p = model.predict(df.iloc[bl])
        blend_preds.append(p)
        singles[name] = cosine_similarity(y[bl], p)
        log.info("  %-9s blend window cosine %+.5f", name, singles[name])

    ens = CosineOptimalEnsemble(model_names=list(models)).fit(
        np.column_stack(blend_preds), y[bl])
    weights = ens.weight_map()
    stack = np.column_stack(blend_preds)
    ens_score = cosine_similarity(y[bl], ens.predict(stack))
    equal = cosine_similarity(y[bl], stack @ np.full(stack.shape[1], 1 / stack.shape[1]))
    log.info("  ensemble  blend window cosine %+.5f   weights %s", ens_score,
             {k: round(v, 4) for k, v in weights.items()})
    log.info("  equal weights would give        %+.5f  (difference %+.5f)",
             equal, ens_score - equal)
    log.info("NOTE the blend window is where the weights were fitted, so %+.5f is "
             "in-sample for the blend and is NOT an out-of-sample estimate", ens_score)

    out_dir = Path(cfg.paths.features)
    model_dir = Path(cfg.paths.data_root) / "models" / "shipped"
    # Store the WRAPPERS, not the inner estimators. RidgeModel carries the median
    # imputation that bare sklearn Ridge has no idea about (see src/models/base.py) -
    # unwrapping it made predict() fail on the first NaN in the test set. The wrappers
    # already share one predict(df) contract, which is the whole point of having it.
    inner = {name: (getattr(m, "booster_", None) or getattr(m, "model", m))
             for name, m in models.items()}
    save_bundle(
        model_dir, model=inner["lightgbm"], kind="lightgbm",
        features=feature_columns(df), name="ensemble", version=version,
        metrics={"train_months": f"0-{TRAIN_END}", "stop_month": STOP_MONTH,
                 "blend_months": f"{BLEND_MONTHS[0]}-{BLEND_MONTHS[1]}",
                 "blend_in_sample_cosine": round(ens_score, 6),
                 **{f"blend_{k}": round(v, 6) for k, v in singles.items()}},
    )
    import joblib

    joblib.dump({"models": models, "weights": weights,
                 "features": feature_columns(df)}, model_dir / "ensemble.joblib")

    meta = {"train_months": [0, TRAIN_END], "stop_month": STOP_MONTH,
            "blend_months": list(BLEND_MONTHS), "weights": weights,
            "singles_blend": singles, "ensemble_blend": ens_score,
            "equal_weight_blend": equal,
            "months_used": TRAIN_END + 1, "months_used_by_submitted_artefact": 64}
    (out_dir / "ship_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log.info("artefact written: %s", model_dir)
    return meta


def predict_test(*, out_name: str = "submission_v2.csv") -> Path:
    """Score the test set with the shipped ensemble and write a submission."""
    import joblib
    import pandas as pd
    import pyarrow.parquet as pq

    cfg = load_config()
    model_dir = Path(cfg.paths.data_root) / "models" / "shipped"
    bundle = joblib.load(model_dir / "ensemble.joblib")
    features, weights = bundle["features"], bundle["weights"]

    feat = Path(cfg.paths.features)
    test = pq.read_table(feat / "dataset_test.parquet",
                         columns=["sample_id"] + features).to_pandas()
    for c in features:
        test[c] = test[c].astype("float32")

    stacked = np.column_stack([
        bundle["models"][name].predict(test[features]) for name in weights
    ])
    pred = stacked @ np.array([weights[name] for name in weights], dtype=float)

    sub = pd.DataFrame({"sample_id": test["sample_id"].to_numpy(), "prediction": pred})
    template = pd.read_csv(Path(cfg.paths.raw) / "submission.csv")
    if set(sub.sample_id) != set(template.sample_id):
        raise SystemExit("sample_id set does not match the submission template")
    sub = sub.sort_values("sample_id", ignore_index=True)

    path = feat / out_name
    sub.to_csv(path, index=False)
    log.info("[submission] %s | %s rows | pred std %.6f | mean %.2e",
             path, f"{len(sub):,}", sub.prediction.std(), sub.prediction.mean())
    return path


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Build and score the shippable ensemble")
    ap.add_argument("--rounds", type=int, default=2000)
    ap.add_argument("--early-stopping", type=int, default=100)
    ap.add_argument("--predict", action="store_true",
                    help="skip training; score the test set with the saved artefact")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if args.predict:
        predict_test()
    else:
        build(rounds=args.rounds, early_stopping=args.early_stopping)
        predict_test()


if __name__ == "__main__":
    main()

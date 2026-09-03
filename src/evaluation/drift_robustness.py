"""Does drift actually predict degradation?

The leaderboard came in at 0.128 against a hold-out of 0.152. Notebook 04's spread-regime
model explained only a quarter of that gap, and notebook 03 nominated the high-drift rate
features as the leading suspect for the rest.

Nominating a suspect is not evidence. The obvious test - retrain without them and submit -
costs a submission and returns exactly one bit. This module tests the MECHANISM instead,
using the training data alone.

THE DESIGN

If high-drift features are a liability under distribution shift, then dropping them should
help MORE when the gap between training and evaluation is LARGER. The training set spans
71 months, so that gap can be varied directly:

    train on months 0-34, fixed          <- never changes, so only the gap varies
    evaluate on 36-40, 42-46, ... 66-70  <- increasing distance into the future

and the quantity of interest is not either score but their DIFFERENCE:

    lift(gap) = cosine(pruned) - cosine(full)

A drift mechanism predicts lift rises with gap. A flat line falsifies it - and would mean
the drift ranking, however well measured, does not identify the features that hurt.

WHY THE COMPARISON IS PAIRED

Both feature sets see identical rows, identical folds and identical seeds; only the column
list differs. Run-to-run noise is therefore shared and cancels in the difference, which
matters because the effect being looked for (~0.005) is close to the fold-to-fold std
(0.004). Seeds are averaged for the same reason.

NO EARLY STOPPING. Stopping on the evaluation block would tune to the thing being measured
and manufacture the trend. Every model trains for a fixed number of rounds.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_config
from src.evaluation.metrics import cosine_similarity
from src.models.base import feature_columns
from src.models.train import load_dataset

log = logging.getLogger(__name__)

TRAIN_MONTHS = (0, 34)
EVAL_BLOCKS = [(36, 40), (42, 46), (48, 52), (54, 58), (60, 64), (66, 70)]
PARAMS = {
    "objective": "regression", "metric": "None", "learning_rate": 0.05,
    "num_leaves": 127, "min_data_in_leaf": 300, "feature_fraction": 0.8,
    "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 5.0,
    "max_bin": 127, "verbosity": -1, "num_threads": 0,
}


def drift_ranking(path: str | Path | None = None) -> pd.DataFrame:
    cfg = load_config()
    path = Path(path or Path(cfg.paths.features) / "drift_report.csv")
    d = pd.read_csv(path)
    d["abs_shift"] = d["shift"].abs()
    return d.sort_values("abs_shift", ascending=False).reset_index(drop=True)


def pruned_columns(all_cols: list[str], drift: pd.DataFrame, threshold: float) -> list[str]:
    """Feature columns whose absolute standardised mean shift is below `threshold`."""
    drop = set(drift.loc[drift.abs_shift >= threshold, "feature"])
    return [c for c in all_cols if c not in drop]


def run(
    *,
    threshold: float = 0.2,
    seeds: tuple[int, ...] = (0, 1, 2),
    rounds: int = 300,
    df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    import lightgbm as lgb

    cfg = load_config()
    if df is None:
        df = load_dataset("train")
    drift = drift_ranking()
    all_cols = feature_columns(df)
    keep = pruned_columns(all_cols, drift, threshold)
    log.info("full: %d features | pruned (|shift| < %.2f): %d features | dropped %d",
             len(all_cols), threshold, len(keep), len(all_cols) - len(keep))

    months = df["month"].to_numpy()
    y = df["target"].to_numpy()
    tr = np.where((months >= TRAIN_MONTHS[0]) & (months <= TRAIN_MONTHS[1]))[0]
    log.info("train months %d-%d: %s rows", *TRAIN_MONTHS, f"{len(tr):,}")

    sets = {"full": all_cols, "pruned": keep}
    preds: dict[tuple[str, int], np.ndarray] = {}
    for name, cols in sets.items():
        for seed in seeds:
            booster = lgb.train({**PARAMS, "seed": seed}, lgb.Dataset(df.iloc[tr][cols],
                                label=y[tr]), num_boost_round=rounds)
            preds[(name, seed)] = booster.predict(df[cols])
            log.info("  trained %-6s seed %d", name, seed)

    rows = []
    for lo, hi in EVAL_BLOCKS:
        va = np.where((months >= lo) & (months <= hi))[0]
        gap = lo - TRAIN_MONTHS[1]
        rec = {"block": f"{lo}-{hi}", "gap_months": gap, "n": len(va)}
        for name in sets:
            s = [cosine_similarity(y[va], preds[(name, seed)][va]) for seed in seeds]
            rec[name] = float(np.mean(s))
            rec[f"{name}_std"] = float(np.std(s))
        rec["lift"] = rec["pruned"] - rec["full"]
        rows.append(rec)
        log.info("  %-7s gap %2d months  full %+.5f  pruned %+.5f  lift %+.5f",
                 rec["block"], gap, rec["full"], rec["pruned"], rec["lift"])

    out = pd.DataFrame(rows)
    slope, corr = trend(out)
    log.info("\n%s", out.to_string(index=False, float_format=lambda v: f"{v:,.5f}"))
    log.info("lift vs gap: slope %+.6f per month, Pearson r %+.3f", slope, corr)

    dst = Path(cfg.paths.features)
    out.to_csv(dst / "drift_robustness.csv", index=False)
    (dst / "drift_robustness_meta.json").write_text(json.dumps({
        "threshold": threshold, "seeds": list(seeds), "rounds": rounds,
        "n_features_full": len(all_cols), "n_features_pruned": len(keep),
        "train_months": list(TRAIN_MONTHS), "slope_per_month": slope, "pearson_r": corr,
    }, indent=2), encoding="utf-8")
    return out


def trend(out: pd.DataFrame) -> tuple[float, float]:
    """Slope of lift against gap, and their correlation."""
    g, lift = out["gap_months"].to_numpy(float), out["lift"].to_numpy()
    slope = float(np.polyfit(g, lift, 1)[0])
    corr = float(np.corrcoef(g, lift)[0, 1])
    return slope, corr


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Test whether drift predicts degradation")
    ap.add_argument("--threshold", type=float, default=0.2)
    ap.add_argument("--rounds", type=int, default=300)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    run(threshold=args.threshold, seeds=tuple(args.seeds), rounds=args.rounds)


if __name__ == "__main__":
    main()

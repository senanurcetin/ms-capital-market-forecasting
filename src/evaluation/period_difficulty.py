"""Was the hold-out an unusually easy period?

Three explanations for the leaderboard shortfall have now been rejected: high-drift
features, decay with elapsed time, and the test set being categorically different (it is
in fact LESS distinguishable from the last training months than two adjacent training
blocks are from each other). What remains is the estimator itself.

The hold-out is a single contiguous stretch - months 65-70 - and it was chosen for being
last, not for being typical. If those particular months happen to be favourable, the
hold-out overstates what a fresh period will produce, and no amount of methodological care
elsewhere would reveal it: the number is measured correctly on a sample of one period.

THE MEASUREMENT

Hold the model FIXED (trained once on months 0-34) and score it on every later period in
turn. Training set, features, seeds and rounds are identical across blocks, so the only
thing varying is which months are being predicted. Whatever spread appears is period
difficulty, not model variance.

The question is then simply where months 65-70 fall in that distribution - and whether the
gap between them and a typical period is big enough to matter next to the 0.024 that needs
explaining.

Absolute levels here are lower than the reported hold-out score because the model trains on
half the data. What transfers is the RELATIVE difficulty of periods, which is what the
correction needs.
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
HOLDOUT = (65, 70)
BLOCK = 3
FIRST_EVAL = 36
LAST_MONTH = 70
PARAMS = {
    "objective": "regression", "metric": "None", "learning_rate": 0.05,
    "num_leaves": 127, "min_data_in_leaf": 300, "feature_fraction": 0.8,
    "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 5.0,
    "max_bin": 127, "verbosity": -1, "num_threads": 0,
}


def blocks(first: int = FIRST_EVAL, last: int = LAST_MONTH, size: int = BLOCK):
    return [(lo, min(lo + size - 1, last)) for lo in range(first, last + 1, size)]


def run(*, seeds: tuple[int, ...] = (0, 1, 2), rounds: int = 300) -> pd.DataFrame:
    import lightgbm as lgb

    cfg = load_config()
    df = load_dataset("train")
    cols = feature_columns(df)
    months, y = df["month"].to_numpy(), df["target"].to_numpy()

    tr = np.where((months >= TRAIN_MONTHS[0]) & (months <= TRAIN_MONTHS[1]))[0]
    log.info("fixed training set: months %d-%d, %s rows, %d features",
             *TRAIN_MONTHS, f"{len(tr):,}", len(cols))

    preds = {}
    for seed in seeds:
        booster = lgb.train({**PARAMS, "seed": seed},
                            lgb.Dataset(df.iloc[tr][cols], label=y[tr]),
                            num_boost_round=rounds)
        preds[seed] = booster.predict(df[cols])
        log.info("  trained seed %d", seed)

    rows = []
    for lo, hi in blocks():
        m = np.where((months >= lo) & (months <= hi))[0]
        if len(m) == 0:
            continue
        s = [cosine_similarity(y[m], preds[seed][m]) for seed in seeds]
        rows.append({"block": f"{lo}-{hi}", "lo": lo, "hi": hi, "n": len(m),
                     "cosine": float(np.mean(s)), "seed_std": float(np.std(s))})
        log.info("  months %2d-%2d  n=%6d  cosine %+.5f", lo, hi, len(m), rows[-1]["cosine"])

    out = pd.DataFrame(rows)

    # The hold-out span, scored the same way, against everything that is not it.
    hm = np.where((months >= HOLDOUT[0]) & (months <= HOLDOUT[1]))[0]
    holdout_cos = float(np.mean([cosine_similarity(y[hm], preds[s][hm]) for s in seeds]))
    others = out[~((out.lo >= HOLDOUT[0]) & (out.hi <= HOLDOUT[1]))]
    typical = float(others.cosine.median())
    # Percentile of the HOLD-OUT score among the block scores - not of the last
    # block, which is a different and useless number.
    pct = float((out.cosine < holdout_cos).mean())

    ratio = holdout_cos / typical
    log.info("")
    log.info("hold-out months %d-%d      %+.5f", *HOLDOUT, holdout_cos)
    log.info("median of other blocks    %+.5f", typical)
    log.info("hold-out is %.1f%% above a typical period", (ratio - 1) * 100)
    log.info("hold-out sits above %.0f%% of all blocks (%d of %d)",
             pct * 100, int((out.cosine < holdout_cos).sum()), len(out))

    # What that implies for the reported score.
    reported, actual = 0.15171, 0.128
    implied = reported / ratio
    log.info("")
    log.info("reported hold-out score            %+.5f", reported)
    log.info("de-biased to a typical period      %+.5f", implied)
    log.info("actual leaderboard                 %+.5f", actual)
    log.info("period luck explains %.0f%% of the %.5f gap",
             (reported - implied) / (reported - actual) * 100, reported - actual)

    dst = Path(cfg.paths.features)
    out.to_csv(dst / "period_difficulty.csv", index=False)
    (dst / "period_difficulty_meta.json").write_text(json.dumps({
        "train_months": list(TRAIN_MONTHS), "holdout": list(HOLDOUT), "block_size": BLOCK,
        "seeds": list(seeds), "rounds": rounds, "holdout_cosine": holdout_cos,
        "typical_cosine": typical, "ratio": ratio, "holdout_percentile": pct,
        "reported": reported, "debiased": implied, "actual": actual,
        "share_of_gap": (reported - implied) / (reported - actual),
    }, indent=2), encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Is the hold-out period unusually easy?")
    ap.add_argument("--rounds", type=int, default=300)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    run(seeds=tuple(args.seeds), rounds=args.rounds)


if __name__ == "__main__":
    main()

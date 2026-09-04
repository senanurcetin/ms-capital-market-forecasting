"""What does the submitted model lose by stopping at month 63?

The model behind the 0.128 leaderboard score was trained on months 0-63, early-stopped on
64, with 65-70 held out. That protocol is correct for MEASURING - and wrong for shipping.
A hold-out has done its job once it has been read; carrying it through to the final
artefact throws away 7 of 71 months, and the most recent ones at that, which sit closest
to the test period.

This measures the cost, using training labels only - no submission required.

THE DESIGN, AND THE CONFOUND IT HAS TO HANDLE

Evaluate everything on the same block (months 65-70) and vary where training STOPS:

    cumulative   0-51, 0-57, 0-63     more months AND more recent months
    fixed window 0-51, 6-57, 12-63    52 months each, only the endpoint moves

The cumulative arm answers the decision directly, because shipping a model trained
through month 70 changes volume and recency together, exactly as these do. The fixed
window arm exists to say WHY: if it shows the same slope, recency is doing the work; if
it is flat, the gain is simply more data.

No early stopping anywhere. Each arm would need its own validation month, and a stopping
rule that differs by arm is not a controlled comparison - so every model trains for the
same fixed number of rounds.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_config
from src.evaluation.metrics import cosine_similarity
from src.models.base import feature_columns
from src.models.lightgbm_model import DEFAULT_PARAMS
from src.models.train import load_dataset

log = logging.getLogger(__name__)

EVAL = (65, 70)
CUMULATIVE = [(0, 51), (0, 57), (0, 63)]
FIXED_WINDOW = [(0, 51), (6, 57), (12, 63)]
ROUNDS = 800


def _fit_score(df: pd.DataFrame, lo: int, hi: int, ev: np.ndarray, seed: int) -> float:
    import lightgbm as lgb

    cols = feature_columns(df)
    months, y = df["month"].to_numpy(), df["target"].to_numpy()
    tr = np.flatnonzero((months >= lo) & (months <= hi))
    assert months[tr].max() < EVAL[0], "training window must not touch the evaluation block"
    booster = lgb.train({**DEFAULT_PARAMS, "seed": seed, "bagging_seed": seed,
                         "feature_fraction_seed": seed},
                        lgb.Dataset(df.iloc[tr][cols], label=y[tr]),
                        num_boost_round=ROUNDS)
    return cosine_similarity(y[ev], booster.predict(df.iloc[ev][cols]))


def run(*, seeds: tuple[int, ...] = (0, 1), rounds: int = ROUNDS) -> pd.DataFrame:
    global ROUNDS
    ROUNDS = rounds
    cfg = load_config()
    df = load_dataset("train")
    months = df["month"].to_numpy()
    ev = np.flatnonzero((months >= EVAL[0]) & (months <= EVAL[1]))
    log.info("evaluating on months %d-%d (%s rows), %d rounds, seeds %s",
             *EVAL, f"{len(ev):,}", rounds, list(seeds))

    rows = []
    for arm, windows in (("cumulative", CUMULATIVE), ("fixed_window", FIXED_WINDOW)):
        for lo, hi in windows:
            t0 = time.perf_counter()
            s = [_fit_score(df, lo, hi, ev, seed) for seed in seeds]
            gap = EVAL[0] - hi          # months between the end of training and evaluation
            rows.append({"arm": arm, "window": f"{lo}-{hi}", "months": hi - lo + 1,
                         "gap_to_eval": gap, "cosine": float(np.mean(s)),
                         "seed_std": float(np.std(s)),
                         "n_train": int(((months >= lo) & (months <= hi)).sum())})
            log.info("  %-12s train %5s (%2d months, gap %2d)  cosine %+.5f  [%.0f s]",
                     arm, rows[-1]["window"], rows[-1]["months"], gap, rows[-1]["cosine"],
                     time.perf_counter() - t0)

    out = pd.DataFrame(rows)
    cum = out[out.arm == "cumulative"].sort_values("gap_to_eval")
    fix = out[out.arm == "fixed_window"].sort_values("gap_to_eval")

    # Cost per month of staleness, from each arm.
    cum_slope = float(np.polyfit(cum.gap_to_eval, cum.cosine, 1)[0])
    fix_slope = float(np.polyfit(fix.gap_to_eval, fix.cosine, 1)[0])

    log.info("\n%s", out.to_string(index=False, float_format=lambda v: f"{v:,.5f}"))
    log.info("cumulative   : %+.6f cosine per month of staleness", cum_slope)
    log.info("fixed window : %+.6f cosine per month of staleness", fix_slope)

    # The submitted model stopped at 63; a shipped model would stop at 70.
    freshest = float(cum.loc[cum.gap_to_eval.idxmin(), "cosine"])
    stalest = float(cum.loc[cum.gap_to_eval.idxmax(), "cosine"])
    per_month = (freshest - stalest) / (cum.gap_to_eval.max() - cum.gap_to_eval.min())
    projected = per_month * 7          # months 64-70, discarded by the submitted model
    log.info("")
    log.info("measured: %+.5f -> %+.5f moving the training end %d months closer",
             stalest, freshest, int(cum.gap_to_eval.max() - cum.gap_to_eval.min()))
    log.info("extrapolating the same rate over the 7 discarded months (64-70): %+.5f",
             projected)

    dst = Path(cfg.paths.features)
    out.to_csv(dst / "recency.csv", index=False)
    (dst / "recency_meta.json").write_text(json.dumps({
        "eval_months": list(EVAL), "rounds": rounds, "seeds": list(seeds),
        "cumulative_slope_per_month": cum_slope, "fixed_window_slope_per_month": fix_slope,
        "per_month_gain": float(per_month), "projected_gain_7_months": float(projected),
    }, indent=2), encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Cost of training through month 63, not 70")
    ap.add_argument("--rounds", type=int, default=ROUNDS)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    run(seeds=tuple(args.seeds), rounds=args.rounds)


if __name__ == "__main__":
    main()

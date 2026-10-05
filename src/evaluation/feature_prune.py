"""Does dropping the near-duplicate features cost anything?

`feature_audit` found that 28 of the 292 features are copies of another one (31 pairs above
|r| 0.999), so the honest count is 264. It also said what that does and does not mean:
nothing already measured is invalidated, and "whether pruning would raise the score is a
separate question ... that would need measuring rather than assuming". This is that
measurement.

THE TEST

Identical folds, rows and seeds; only the column list differs. From every group of
near-duplicates the FIRST column (in table order) is kept and the rest dropped. That rule
is arbitrary on purpose: choosing the member to keep by its importance would use the
target, and a duplicate carries the same information whichever copy survives.

THE DECISION RULE, fixed before any run so it cannot be bent afterwards

The question is non-inferiority, not improvement. With a fold-to-fold std of 0.0041 an
improvement of the size pruning could plausibly give is unmeasurable, and this project has
five forecasts that overshot to show what claiming one costs. What can be asked is whether
pruning LOSES anything that matters:

  CI lower bound  > -0.0010   pruning is free: adopt the smaller set for speed and size,
                              and claim nothing about the score
  CI lower bound <= -0.0010   pruning costs something, or the test cannot rule it out:
                              keep all 292

-0.0010 is about the size of the ensemble's gain, the smallest effect this project has
measured. A loss that small is below what an external score would show.

A positive gain, if it appears, is evidence about WHICH feature set to prefer and nothing
more - the same rule notebook 05 ends on.

Nothing is reported from this module until it has been run on the full data; the figures it
writes (feature_prune.csv, feature_prune_meta.json) are the only source for any claim.
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
from src.evaluation.feature_audit import NEAR_DUPLICATE, correlation_pairs, duplicate_groups
from src.evaluation.shape_gain import FOLDS, ROUNDS, cv
from src.models.base import feature_columns

log = logging.getLogger(__name__)

# A loss smaller than this is below what an external score would show (see module docstring).
NON_INFERIORITY_MARGIN = 0.0010


def features_to_drop(pairs: pd.DataFrame, cols: list[str],
                     threshold: float = NEAR_DUPLICATE) -> list[str]:
    """Every column that is a near-duplicate of an EARLIER one, in table order.

    From each duplicate group the first column survives. Choosing the survivor by feature
    importance would feed the target into a selection step; the copies carry the same
    information, so the choice is made by position and is therefore free of it.
    """
    drop: list[str] = []
    for group in duplicate_groups(pairs, cols, threshold):
        drop.extend(group[1:])
    return sorted(drop, key=cols.index)


def verdict(diffs: np.ndarray, margin: float = NON_INFERIORITY_MARGIN) -> dict:
    """Apply the non-inferiority rule to paired per-fold differences (pruned - full)."""
    diffs = np.asarray(diffs, dtype=float)
    gain = float(diffs.mean())
    se = float(diffs.std(ddof=1) / np.sqrt(len(diffs))) if len(diffs) > 1 else float("inf")
    lo, hi = gain - 1.96 * se, gain + 1.96 * se
    free = lo > -margin
    return {
        "paired_gain": gain, "se": se, "ci_low": lo, "ci_high": hi,
        "n_comparisons": int(len(diffs)), "improved": int((diffs > 0).sum()),
        "margin": margin, "pruning_is_free": bool(free),
        "verdict": ("pruning is free: adopt the smaller set, claim nothing about the score"
                    if free else "keep all features: pruning costs something or cannot "
                                 "be ruled out as costing it"),
    }


def run(*, seeds: tuple[int, ...] = (0, 1), folds: int = FOLDS,
        n_sample: int = 150_000, threshold: float = NEAR_DUPLICATE) -> dict:
    from src.models.train import load_dataset

    cfg = load_config()
    df = load_dataset("train")
    cols = feature_columns(df)

    pairs = correlation_pairs(df.sample(n=min(n_sample, len(df)), random_state=0), cols)
    drop = features_to_drop(pairs, cols, threshold)
    kept = [c for c in cols if c not in set(drop)]
    log.info("full %d features | dropping %d near-duplicates | pruned %d",
             len(cols), len(drop), len(kept))
    log.info("dropped: %s", drop)

    rows = []
    for seed in seeds:
        t0 = time.perf_counter()
        full = cv(df, cols, seed=seed, folds=folds)
        pruned = cv(df, kept, seed=seed, folds=folds)
        for i, (f, p) in enumerate(zip(full, pruned, strict=True)):
            rows.append({"seed": seed, "fold": i, "full": f, "pruned": p, "diff": p - f})
        log.info("seed %d  full %+.5f  pruned %+.5f  diff %+.5f   [%.0f min]", seed,
                 np.mean(full), np.mean(pruned), np.mean(pruned) - np.mean(full),
                 (time.perf_counter() - t0) / 60)

    out = pd.DataFrame(rows)
    res = verdict(out["diff"].to_numpy())
    log.info("\n%s", out.to_string(index=False, float_format=lambda v: f"{v:,.5f}"))
    log.info("paired gain %+.5f  95%% CI [%+.5f, %+.5f]  margin -%.4f  improved %d of %d",
             res["paired_gain"], res["ci_low"], res["ci_high"], res["margin"],
             res["improved"], res["n_comparisons"])
    log.info("VERDICT: %s", res["verdict"])

    res.update({"seeds": list(seeds), "folds": folds, "rounds": ROUNDS,
                "n_full": len(cols), "n_pruned": len(kept), "dropped": drop,
                "threshold": threshold})
    dst = Path(cfg.paths.features)
    out.to_csv(dst / "feature_prune.csv", index=False)
    (dst / "feature_prune_meta.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    return res


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Does dropping near-duplicate features cost anything?")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--folds", type=int, default=FOLDS)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    run(seeds=tuple(args.seeds), folds=args.folds)


if __name__ == "__main__":
    main()

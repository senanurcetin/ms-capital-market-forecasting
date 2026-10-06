"""Is `sample_id` a time axis in train, and is it one in test?

WHY THIS WAS MEASURED. The README says the one context the 292 features cannot see is "what the
market is doing around each sample", and the regime-scaling correction in `prediction_geometry`
builds exactly that from the neighbouring samples' volatility. Both quietly assume that
neighbouring `sample_id`s are neighbours in time. In train they are: `sample_id` is chronological.
Whether they are in the *test* set decides whether any neighbour-derived feature, correction or
context can be computed there at all, and nothing in the repo had checked.

THE MEASUREMENT. For a few slow market-state signals (volatility, trade intensity, spread), cut the
sample sequence into blocks of k consecutive ids, average the log of the signal in each block, and
report the lag-1 autocorrelation of those block means. A sequence with time structure has smooth
regimes, so neighbouring blocks resemble each other (close to +1); a sequence in random order does
not (close to 0, with a standard error of about 1 / sqrt(number of blocks)).

Then the same thing with each month's mean removed, because most neighbouring blocks share a month:
a high plain autocorrelation can come entirely from differences *between* months and says nothing
about order *within* one.

The answer in this repo's data is in `results/split_order.json`.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import load_config

log = logging.getLogger(__name__)

SIGNALS = ("mkt_mid_std_60s", "txn_intensity_60s", "mkt_rel_spread_mean_5s")
BLOCK_SIZES = (10, 100, 1000, 10000)


def block_autocorr(values: pd.Series, block: int) -> tuple[float, int]:
    """Lag-1 autocorrelation of the block means of log(values), and the number of blocks.

    Non-positive values are missing (a log of them is undefined, and a zero volatility is a flat
    book rather than a measurement). Blocks with no finite value are dropped, not zero-filled.
    """
    s = np.log(values.where(values > 0))
    means = s.groupby(np.arange(len(s)) // block).mean().dropna()
    if len(means) < 3:
        return float("nan"), len(means)
    return float(means.autocorr(1)), int(len(means))


def within_group_block_autocorr(values: pd.Series, groups: pd.Series, block: int) -> tuple[float, int]:
    """The same measure after removing each group's mean: structure *inside* a month, if any.

    Between-month differences alone make neighbouring blocks alike (most of them share a month),
    so a high plain autocorrelation does not by itself show that samples are ordered within a month.

    Subtracting a group mean is not free: the blocks of one group then sum to zero, which forces a
    NEGATIVE autocorrelation even in a sequence with no order at all, about -1 / (blocks per group - 1).
    Read the result against that, not against zero (`expected_if_unordered` in `within_month`).
    """
    s = np.log(values.where(values > 0))
    resid = s - s.groupby(groups).transform("mean")
    means = resid.groupby(np.arange(len(resid)) // block).mean().dropna()
    if len(means) < 3:
        return float("nan"), len(means)
    return float(means.autocorr(1)), int(len(means))


def measure(frames: dict[str, pd.DataFrame]) -> list[dict]:
    rows = []
    for split, df in frames.items():
        for signal in SIGNALS:
            for block in BLOCK_SIZES:
                ac, n = block_autocorr(df[signal], block)
                rows.append({"split": split, "signal": signal, "block": block, "n_blocks": n,
                             "autocorr": ac, "noise_se": 1 / np.sqrt(n) if n else float("nan")})
    return rows


def summarise(rows: list[dict]) -> dict[str, Any]:
    """The one-line comparison: strongest block size per split, averaged over the signals."""
    t = pd.DataFrame(rows)
    out: dict[str, Any] = {}
    for split in t["split"].unique():
        sub = t[(t["split"] == split) & (t["block"] == 1000)]
        out[split] = {"autocorr_at_1000": float(sub["autocorr"].mean()),
                      "n_blocks_at_1000": int(sub["n_blocks"].iloc[0])}
    train, test = out["train"]["autocorr_at_1000"], out["test"]["autocorr_at_1000"]
    se = 1 / np.sqrt(out["test"]["n_blocks_at_1000"])
    out["test_is_in_time_order"] = bool(test > 3 * se)
    out["train_is_in_time_order"] = bool(train > 3 * (1 / np.sqrt(out["train"]["n_blocks_at_1000"])))
    out["verdict"] = (
        "neighbouring sample_ids are neighbours in time in train but not in test: a feature or "
        "correction built from neighbouring samples cannot be computed on the test set"
        if out["train_is_in_time_order"] and not out["test_is_in_time_order"] else
        "both splits are in time order" if out["train_is_in_time_order"] else
        "neither split is in time order")
    return out


def within_month(train: pd.DataFrame, block: int = 1000) -> dict:
    """Mean over the signals of the within-month residual autocorrelation, for the train split."""
    vals = [within_group_block_autocorr(train[c], train["month"], block) for c in SIGNALS]
    per_month = train.groupby("month").size().mean() / block
    return {"autocorr_at_1000": float(np.mean([v[0] for v in vals])),
            "n_blocks_at_1000": int(vals[0][1]),
            "expected_if_unordered": float(-1 / (per_month - 1))}


def run(out_dir: Path | None = None) -> dict:
    import pyarrow.parquet as pq

    cfg = load_config()
    feats = Path(cfg.paths.features)
    train = pq.read_table(feats / "dataset_train.parquet", columns=["month", *SIGNALS]).to_pandas()
    test = pq.read_table(feats / "dataset_test.parquet", columns=list(SIGNALS)).to_pandas()
    rows = measure({"train": train[list(SIGNALS)], "test": test})
    result = {"signals": list(SIGNALS), "block_sizes": list(BLOCK_SIZES), **summarise(rows),
              "train_within_month": within_month(train),
              "samples_per_train_month": float(train.groupby("month").size().mean()),
              "test_samples": int(len(test))}
    out_dir = out_dir or Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "split_order.csv", index=False, float_format="%.5f")
    (out_dir / "split_order.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Is sample_id a time axis in train and in test?")
    ap.add_argument("--out", default="results")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    print(json.dumps(run(Path(args.out)), indent=2))


if __name__ == "__main__":
    main()

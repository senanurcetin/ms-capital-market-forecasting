"""Synthetic data with the real schema, so the project can be run by anyone.

THE PROBLEM THIS SOLVES: the real pipeline needs Kaggle credentials, a GCP project,
~20 GB of disk and hours of upload. That makes the repository unrunnable for a reader,
which is a poor property for a project meant to be read.

`make demo` generates data here instead and drives the genuine ingestion, training,
evaluation and serving code end to end in about three minutes.

WHAT IS FAITHFUL, AND WHAT IS NOT
  Faithful - deliberately, because these are the quirks the pipeline exists to handle:
    * one Arrow record batch per file, compressed (the constraint behind column groups)
    * a 600 s market window against 60 s for order/transaction
    * price = 0 as an empty-level sentinel, always paired with volume = 0
    * prices normalised around mid = 1.0
    * side 0 = BID/BUY, 1 = ASK/SELL; order_action 0 = NEW, 1 = CANCEL
    * seconds_before_predict descending within a sample
  Not faithful:
    * the signal is planted and far stronger than reality, so a demo model trained on
      a few thousand samples produces a visibly non-zero score
    * volumes and arrival processes are crude
    * NO conclusion in this repository is drawn from synthetic data

The feature table's COLUMN NAMES are read from the real SQL generators rather than
written out here, so the demo schema cannot drift away from the production one.
"""
from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.feather as feather

from src.config import load_config

log = logging.getLogger(__name__)

EMPTY_BID_RATE = 0.0011   # measured on the real data: 133,781 / 221.8M
EMPTY_ASK_RATE = 0.0041   # 898,820 / 221.8M


def _sample_lengths(rng: np.random.Generator, n: int, mean: float, cap: int) -> np.ndarray:
    """Row counts per sample: a long-tailed draw that never exceeds the ceiling."""
    lengths = rng.poisson(mean, n) + 1
    return np.minimum(lengths, cap)


def make_market(rng: np.random.Generator, n_samples: int, window: float) -> pa.Table:
    lengths = _sample_lengths(rng, n_samples, mean=176, cap=212)
    total = int(lengths.sum())
    sample_id = np.repeat(np.arange(n_samples, dtype=np.int32), lengths)

    # Descending within each sample, as in the real files.
    secs = np.concatenate([
        np.sort(rng.uniform(0, window, k))[::-1] for k in lengths
    ]).astype(np.float32)

    mid = (1.0 + rng.normal(0, 0.004, total)).astype(np.float32)
    # A floor on the half-spread. Without it the draw can land near zero and produce
    # bid >= ask, and the real data contains EXACTLY ZERO genuinely crossed books - a
    # property the pipeline relies on when it treats every apparent crossing as a
    # sentinel artefact.
    half = np.maximum(np.abs(rng.normal(0.00063, 0.0002, total)), 5e-5).astype(np.float32)
    bid1 = (mid - half).astype(np.float32)
    ask1 = (mid + half).astype(np.float32)
    bv1 = rng.integers(100, 50_000, total).astype(np.int32)
    av1 = rng.integers(100, 50_000, total).astype(np.int32)

    # Empty-level sentinel: price AND volume both zero, exactly as observed.
    empty_bid = rng.random(total) < EMPTY_BID_RATE
    empty_ask = rng.random(total) < EMPTY_ASK_RATE
    bid1[empty_bid] = 0.0
    bv1[empty_bid] = 0
    ask1[empty_ask] = 0.0
    av1[empty_ask] = 0

    bid2 = np.where(bid1 > 0, bid1 - np.abs(rng.normal(0.0004, 0.0002, total)), 0)
    ask2 = np.where(ask1 > 0, ask1 + np.abs(rng.normal(0.0004, 0.0002, total)), 0)

    return pa.table({
        "sample_id": pa.array(sample_id, pa.int32()),
        "seconds_before_predict": pa.array(secs, pa.float32()),
        "transaction_avgprice": pa.array(mid + rng.normal(0, 0.0003, total), pa.float32()),
        "transaction_volume": pa.array(rng.integers(0, 5000, total), pa.int32()),
        "transaction_count": pa.array(rng.integers(0, 40, total), pa.int32()),
        "ask_price_1": pa.array(ask1, pa.float32()),
        "ask_volume_1": pa.array(av1, pa.int32()),
        "bid_price_1": pa.array(bid1, pa.float32()),
        "bid_volume_1": pa.array(bv1, pa.int32()),
        "ask_price_2": pa.array(ask2, pa.float32()),
        "ask_volume_2": pa.array(rng.integers(100, 60_000, total), pa.int32()),
        "bid_price_2": pa.array(bid2, pa.float32()),
        "bid_volume_2": pa.array(rng.integers(100, 60_000, total), pa.int32()),
    })


def make_order(rng: np.random.Generator, n_samples: int, window: float) -> pa.Table:
    lengths = _sample_lengths(rng, n_samples, mean=135, cap=999)
    total = int(lengths.sum())
    sample_id = np.repeat(np.arange(n_samples, dtype=np.int32), lengths)
    secs = np.concatenate([np.sort(rng.uniform(0, window, k))[::-1] for k in lengths])
    side = (rng.random(total) < 0.5).astype(np.int8)          # 0 = BID, 1 = ASK
    action = (rng.random(total) < 0.25).astype(np.int8)       # 0 = NEW, 1 = CANCEL
    # BID orders sit below mid, ASK above - the property that let the encoding be recovered.
    price = np.where(side == 0, 1.0 - np.abs(rng.normal(0.002, 0.001, total)),
                     1.0 + np.abs(rng.normal(0.002, 0.001, total)))
    return pa.table({
        "sample_id": pa.array(sample_id, pa.int32()),
        "seconds_before_predict": pa.array(secs.astype(np.float32), pa.float32()),
        "price": pa.array(price.astype(np.float32), pa.float32()),
        "volume": pa.array(rng.integers(100, 20_000, total), pa.int32()),
        "side": pa.array(side, pa.int8()),
        "order_action": pa.array(action, pa.int8()),
    })


def make_transaction(rng: np.random.Generator, n_samples: int, window: float) -> pa.Table:
    lengths = _sample_lengths(rng, n_samples, mean=83, cap=999)
    total = int(lengths.sum())
    sample_id = np.repeat(np.arange(n_samples, dtype=np.int32), lengths)
    secs = np.concatenate([np.sort(rng.uniform(0, window, k))[::-1] for k in lengths])
    side = (rng.random(total) < 0.5).astype(np.int8)          # 0 = BUY, 1 = SELL
    # Buyer-initiated trades print above mid, seller-initiated below.
    price = np.where(side == 0, 1.0 + np.abs(rng.normal(0.0005, 0.0003, total)),
                     1.0 - np.abs(rng.normal(0.0005, 0.0003, total)))
    return pa.table({
        "sample_id": pa.array(sample_id, pa.int32()),
        "seconds_before_predict": pa.array(secs.astype(np.float32), pa.float32()),
        "price": pa.array(price.astype(np.float32), pa.float32()),
        "volume": pa.array(rng.integers(100, 30_000, total), pa.int32()),
        "side": pa.array(side, pa.int8()),
    })


def make_label(rng: np.random.Generator, n_samples: int, months: int) -> pa.Table:
    per_month = max(1, n_samples // months)
    month = np.minimum(np.arange(n_samples) // per_month, months - 1).astype(np.int16)
    return pa.table({
        "month": pa.array(month, pa.int16()),
        "sample_id": pa.array(np.arange(n_samples, dtype=np.int32), pa.int32()),
        "target": pa.array(rng.normal(0, 0.0026, n_samples).astype(np.float32), pa.float32()),
    })


def write_raw(out_dir: Path, n_samples: int = 4000, months: int = 71, seed: int = 7) -> Path:
    """Write synthetic feather files matching the competition layout."""
    cfg = load_config()
    rng = np.random.default_rng(seed)
    train = out_dir / "train"
    train.mkdir(parents=True, exist_ok=True)

    builders = {
        "market": (make_market, cfg.window.seconds["market"]),
        "order": (make_order, cfg.window.seconds["order"]),
        "transaction": (make_transaction, cfg.window.seconds["transaction"]),
    }
    for name, (fn, window) in builders.items():
        table = fn(rng, n_samples, window)
        path = train / f"{name}.feather"
        # chunksize forces ONE record batch - the constraint the real files impose.
        feather.write_feather(table, path, compression="lz4", chunksize=table.num_rows)
        log.info("[synthetic] %-12s %s rows -> %s", name, f"{table.num_rows:,}", path.name)

    label = make_label(rng, n_samples, months)
    feather.write_feather(label, train / "label.feather", compression="lz4")
    log.info("[synthetic] %-12s %s rows -> label.feather", "label", f"{label.num_rows:,}")
    return out_dir


def feature_names() -> list[str]:
    """Read the feature column names from the REAL SQL generators.

    Deriving them here rather than hardcoding means the demo schema cannot drift
    away from production: if a feature is added or renamed, this follows.
    """
    from src.features import market_features, order_features, transaction_features

    names: list[str] = []
    for module in (market_features, order_features, transaction_features):
        sql = module.build_sql("train")
        found = re.findall(r"\sAS\s+([a-z][a-z0-9_]*)", sql)
        names += [a for a in found if a.startswith(("mkt_", "ord_", "txn_"))]
    return list(dict.fromkeys(names))


def make_feature_table(n_samples: int = 4000, months: int = 71, seed: int = 7):
    """A demo feature table: real column names, synthetic values, a planted signal.

    The signal is deliberately much stronger than reality so that a model trained on a
    few thousand rows scores visibly above zero. No claim in this repository rests on it.
    """
    import pandas as pd

    rng = np.random.default_rng(seed)
    names = feature_names()
    per_month = max(1, n_samples // months)

    data = {
        "sample_id": np.arange(n_samples, dtype=np.int32),
        "month": np.minimum(np.arange(n_samples) // per_month, months - 1).astype(np.int16),
    }
    X = rng.normal(0, 1, (n_samples, len(names))).astype(np.float32)
    for i, name in enumerate(names):
        data[name] = X[:, i]

    # Signal carried by the feature families the real model relies on most.
    drivers = [i for i, n in enumerate(names)
               if "imbalance" in n or n.endswith("_ofi_60s") or n == "mkt_micro_minus_mid_last"]
    drivers = drivers[:8] or [0, 1, 2]
    weights = rng.normal(0, 1, len(drivers))
    signal = X[:, drivers] @ weights
    signal /= signal.std()
    target = 0.4 * signal + rng.normal(0, 1, n_samples)
    data["target"] = (target / target.std() * 0.0026).astype(np.float32)

    df = pd.DataFrame(data)
    # Real feature tables carry NULLs in short windows; keep that property.
    for name in names:
        if name.endswith(("_1s", "_5s")):
            mask = rng.random(n_samples) < 0.4
            df.loc[mask, name] = np.nan
    return df


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Generate synthetic data with the real schema")
    ap.add_argument("--samples", type=int, default=4000)
    ap.add_argument("--months", type=int, default=71)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--raw", action="store_true", help="write raw feather files")
    ap.add_argument("--features", action="store_true", help="write a demo feature table")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    cfg = load_config()
    if args.raw or not args.features:
        write_raw(Path(cfg.paths.raw), args.samples, args.months, args.seed)
    if args.features or not args.raw:
        df = make_feature_table(args.samples, args.months, args.seed)
        dst = Path(cfg.paths.features) / "dataset_train.parquet"
        dst.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(dst, compression="zstd", index=False)
        log.info("[synthetic] feature table %s rows x %s cols -> %s",
                 f"{len(df):,}", df.shape[1], dst)


if __name__ == "__main__":
    main()

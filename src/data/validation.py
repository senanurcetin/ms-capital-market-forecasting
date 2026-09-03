"""Data contracts, expressed as Pandera schemas.

Every constraint here is a fact that was MEASURED during the investigation (notebooks
01-03), not a guess about what the data ought to look like. Writing them down turns each
discovery into something the pipeline enforces rather than something a future reader has
to rediscover.

The interesting ones:

  seconds_before_predict >= 0     look-ahead is structurally impossible; if this ever
                                  fails, the window semantics have changed
  price >= 0                      0 is the empty-level SENTINEL, not a price. Prices are
                                  otherwise 0.909-1.052, so a value in (0, 0.5) would mean
                                  the sentinel convention had changed
  side in {0, 1}                  0 = BID/BUY, 1 = ASK/SELL (recovered by measurement)
  order_action in {0, 1}          0 = NEW, 1 = CANCEL
  rows per sample <= 999          a real ceiling, though it binds for 0.003% of samples

Validation runs on a SAMPLE by default. These files hold hundreds of millions of rows and
a full pass would cost more than it returns; the invariants are structural, so a few
million rows detect a violation just as well as all of them.
"""
from __future__ import annotations

import argparse
import logging

import pandas as pd
from pandera.pandas import Check, Column, DataFrameSchema

from src.config import load_config

log = logging.getLogger(__name__)


def _window(table: str) -> float:
    return float(load_config().window.seconds[table])


def _common(table: str) -> dict[str, Column]:
    """Columns every event table shares."""
    return {
        "sample_id": Column(
            "int32", Check.ge(0), nullable=False,
            description="Sample identifier; train and test have SEPARATE id spaces",
        ),
        "seconds_before_predict": Column(
            "float32",
            [Check.ge(0.0), Check.le(_window(table) + 1e-3)],
            nullable=False,
            description=(
                "Distance BACK from the prediction instant. >= 0 is what makes look-ahead "
                f"structurally impossible; the upper bound is this table's {_window(table):g}s window"
            ),
        ),
    }


def market_schema() -> DataFrameSchema:
    price = [Check.ge(0.0), Check.le(2.0)]
    volume = [Check.ge(0)]
    cols = _common("market")
    cols.update({
        "transaction_avgprice": Column("float32", price, nullable=True),
        "transaction_volume": Column("int32", volume, nullable=True),
        "transaction_count": Column("int32", volume, nullable=True),
    })
    for side in ("ask", "bid"):
        for level in (1, 2):
            cols[f"{side}_price_{level}"] = Column(
                "float32", price, nullable=False,
                description="0 means the level is EMPTY, not a price of zero",
            )
            cols[f"{side}_volume_{level}"] = Column("int32", volume, nullable=False)
    return DataFrameSchema(cols, strict=False, coerce=False, name="market")


def order_schema() -> DataFrameSchema:
    cols = _common("order")
    cols.update({
        "price": Column("float32", [Check.ge(0.0), Check.le(2.0)], nullable=False),
        "volume": Column("int32", Check.ge(0), nullable=False),
        "side": Column("int8", Check.isin([0, 1]), nullable=False,
                       description="0 = BID, 1 = ASK"),
        "order_action": Column("int8", Check.isin([0, 1]), nullable=False,
                               description="0 = NEW, 1 = CANCEL"),
    })
    return DataFrameSchema(cols, strict=False, name="order")


def transaction_schema() -> DataFrameSchema:
    cols = _common("transaction")
    cols.update({
        "price": Column("float32", [Check.ge(0.0), Check.le(2.0)], nullable=False),
        "volume": Column("int32", Check.ge(0), nullable=False),
        "side": Column("int8", Check.isin([0, 1]), nullable=False,
                       description="aggressor side: 0 = BUY, 1 = SELL"),
    })
    return DataFrameSchema(cols, strict=False, name="transaction")


def label_schema() -> DataFrameSchema:
    cfg = load_config()
    return DataFrameSchema({
        "month": Column("int16", [Check.ge(0), Check.lt(cfg.samples["months"])], nullable=False),
        "sample_id": Column("int32", Check.ge(0), nullable=False, unique=True),
        "target": Column(
            "float32", [Check.ge(-0.5), Check.le(0.5)], nullable=False,
            description="A return. Observed range is +-0.084; +-0.5 catches a unit change",
        ),
    }, strict=False, name="label")


SCHEMAS = {
    "market": market_schema,
    "order": order_schema,
    "transaction": transaction_schema,
    "label": label_schema,
}


def check_row_cap(df: pd.DataFrame, table: str) -> None:
    """No sample may exceed the 999-row ceiling in order/transaction."""
    cfg = load_config()
    if table not in cfg.truncation["tables"]:
        return
    cap = int(cfg.truncation["row_cap"])
    worst = df.groupby("sample_id").size().max()
    if worst > cap:
        raise ValueError(f"{table}: a sample has {worst} rows, above the {cap} ceiling")


def check_descending_within_sample(df: pd.DataFrame, n_samples: int = 50) -> None:
    """seconds_before_predict must be non-increasing inside each sample.

    File order is chronological because of this; several features
    (ARRAY_AGG ... ORDER BY seconds) depend on it.
    """
    for sid in df["sample_id"].drop_duplicates().head(n_samples):
        secs = df.loc[df.sample_id == sid, "seconds_before_predict"].to_numpy()
        if (secs[1:] > secs[:-1] + 1e-6).any():
            raise ValueError(f"sample {sid}: seconds_before_predict is not descending")


def validate_raw(split: str = "train", table: str = "market", n_rows: int = 2_000_000) -> dict:
    """Validate the head of a raw feather file against its contract."""
    import pyarrow.feather as feather

    from src.config import raw_path

    path = raw_path(split, table)
    df = feather.read_table(str(path)).to_pandas()
    if n_rows and len(df) > n_rows:
        df = df.head(n_rows)

    SCHEMAS[table]().validate(df, lazy=True)
    if table != "label":
        check_row_cap(df, table)
        check_descending_within_sample(df)
    log.info("[%s/%s] contract OK on %s rows", split, table, f"{len(df):,}")
    return {"split": split, "table": table, "rows_checked": len(df), "ok": True}


def validate_features(split: str = "train") -> dict:
    """Structural checks on the assembled feature table."""
    import numpy as np

    from src.models.train import load_dataset

    cfg = load_config()
    df = load_dataset(split)
    problems = []
    if len(df) != cfg.samples[split]:
        problems.append(f"row count {len(df):,} != {cfg.samples[split]:,}")
    if df["sample_id"].duplicated().any():
        problems.append("duplicate sample_id")
    if not df["sample_id"].is_monotonic_increasing:
        problems.append("not ordered by sample_id")
    numeric = df.select_dtypes("number")
    n_inf = int(np.isinf(numeric.to_numpy()).sum())
    if n_inf:
        problems.append(f"{n_inf} infinite values")
    if split == "train":
        if df["target"].isna().any():
            problems.append("null target")
        months = df["month"]
        if months.min() != 0 or months.max() != cfg.samples["months"] - 1:
            problems.append(f"month range {months.min()}-{months.max()}")
    if problems:
        raise ValueError(f"feature table {split}: " + "; ".join(problems))
    log.info("[features/%s] OK: %s rows x %s columns", split, f"{len(df):,}", df.shape[1])
    return {"split": split, "rows": len(df), "columns": df.shape[1], "ok": True}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Validate data against its contracts")
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--rows", type=int, default=2_000_000, help="rows per raw table")
    ap.add_argument("--skip-raw", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    if not args.skip_raw:
        tables = ["market", "order", "transaction"] + (["label"] if args.split == "train" else [])
        for table in tables:
            validate_raw(args.split, table, args.rows)
    validate_features(args.split)
    log.info("all contracts satisfied")


if __name__ == "__main__":
    main()

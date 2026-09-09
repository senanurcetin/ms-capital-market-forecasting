"""Sequence-shape features: what the aggregation throws away.

Most of the 292 existing features are AGGREGATES over a window - a mean, a rate, a last
value, a standard deviation - and an aggregate is permutation-invariant: shuffle the ~176
snapshots inside a sample and it does not move. Information living in the ORDER of the
book's evolution is largely absent from such a set.

MOSTLY, not entirely - a correction. The first version of this docstring said "every one
of the 292", which was asserted rather than checked and is false. The
`*_delta_300s_vs_600s` family compares nested windows, and that IS a statement about
direction of travel. The audit in src/evaluation/shape_gain.py measured how much overlap
that leaves: `shp_imb_drift` correlates 0.990 with `mkt_depth_imb1_delta_300s_vs_600s`,
and `shp_n_snaps` correlates 1.000 with `mkt_snapshot_rate_600s`, which is a duplicate
that should not have been written. Five of the eighteen exceed 0.9.

The premise still holds for the rest: microstructure theory is mostly about dynamics - how
imbalance builds, whether price moves trend or revert, how quoting bursts cluster - and
little of that survives a GROUP BY.

The leaderboard says the gap is real: 187 teams, median 0.138, this model 0.129. Tuning
bought nothing measurable and the ensemble bought +0.001, so the missing quantity is
information rather than method. Sequence order is the largest identifiable candidate.

WHAT IS COMPUTED

Only quantities that a permutation-invariant statistic CANNOT reproduce:

  path efficiency      |net move| / sum|moves|   - trending vs choppy, same endpoints
  lag-1 autocorrelation of mid returns           - mean reversion vs momentum
  sign persistence     P(same sign as previous)  - runs structure
  RV signature ratio   RV(1 step) / RV(4 step)   - microstructure noise vs true vol
  OLS slope vs time    imbalance, spread, depth  - building or decaying, not just level
  half-window drift    late mean - early mean    - direction of travel
  drawdown / run-up    worst adverse path move   - path extremes, not endpoint extremes
  arrival burstiness   CV of inter-snapshot gaps - clustered vs regular quoting

A mean of imbalance says where the book sat. Its slope says where it was going. Only the
second survives being told the model was wrong about the leaderboard.

THIS IS A DIAGNOSTIC FIRST. The features are added ON TOP of the existing 292 and the
comparison is paired, so the question it answers is narrow and falsifiable: does sequence
order carry signal the aggregates do not already have? If the gain does not clear the
fold-to-fold noise, the sequence-model gate in the plan stays shut - and stays shut on a
measurement rather than on a judgement.
"""
from __future__ import annotations

import argparse
import logging

from src.features.common import feature_table, staged
from src.features.market_features import CLEAN, ROW_DERIVED

log = logging.getLogger(__name__)

# Chronological order inside a sample: seconds_before_predict counts DOWN to the
# prediction instant, so ascending time means DESCENDING seconds.
ORDER = "PARTITION BY sample_id ORDER BY seconds_before_predict DESC"


def build_sql(split: str = "train") -> str:
    src = staged("market", split)
    target = feature_table("shape", split)
    return f"""CREATE OR REPLACE TABLE {target}
CLUSTER BY sample_id AS
WITH clean AS (
  SELECT{CLEAN}  FROM {src}
),
rows_ AS (
  SELECT{ROW_DERIVED}  FROM clean
),
seq AS (
  SELECT
    sample_id,
    -- Carried through because seq2's window functions order by it.
    seconds_before_predict,
    -- t increases chronologically, so a positive slope means "rising towards the
    -- prediction instant" rather than the reverse.
    -seconds_before_predict AS t,
    mid, spread, depth_imb_1, total_depth,
    mid - LAG(mid) OVER ({ORDER}) AS dmid,
    seconds_before_predict
      - LAG(seconds_before_predict) OVER ({ORDER}) AS dt,
    ROW_NUMBER() OVER ({ORDER}) AS rn,
    COUNT(*) OVER (PARTITION BY sample_id) AS n_snaps
  FROM rows_
),
seq2 AS (
  SELECT
    *,
    LAG(dmid) OVER ({ORDER}) AS dmid_lag,
    -- 4-step returns, for the realised-volatility signature ratio.
    IF(MOD(rn, 4) = 0, mid - LAG(mid, 4) OVER ({ORDER}), NULL) AS dmid4
  FROM seq
)
SELECT
  sample_id,

  -- ---- path shape -------------------------------------------------------------
  -- Same start and end, different journey: a trending path scores near 1, a path that
  -- wanders and returns scores near 0. No aggregate of the levels can express this.
  SAFE_DIVIDE(ABS(SUM(dmid)), NULLIF(SUM(ABS(dmid)), 0))       AS shp_mid_path_efficiency,
  SAFE_DIVIDE(MAX(mid) - MIN(mid),
              NULLIF(SUM(ABS(dmid)), 0))                       AS shp_mid_range_over_path,

  -- ---- return dynamics --------------------------------------------------------
  -- CORR returns NaN when a sample's mid never moves (5.0% of samples): the
  -- correlation is undefined, not zero, so it becomes NULL. NaN would survive into
  -- the parquet, break Ridge's imputation and poison any mean computed over it.
  IF(IS_NAN(CORR(dmid, dmid_lag)), NULL,
     CORR(dmid, dmid_lag))                                     AS shp_mid_autocorr1,
  AVG(IF(dmid IS NULL OR dmid_lag IS NULL, NULL,
         IF(SIGN(dmid) = SIGN(dmid_lag) AND dmid != 0, 1.0, 0.0)))
                                                               AS shp_sign_persistence,
  -- Signature ratio. Pure price moves scale with sqrt(time), so this sits near 1;
  -- microstructure noise inflates the fine-grained estimate and pushes it above 1.
  SAFE_DIVIDE(SQRT(SUM(POW(dmid, 2))),
              NULLIF(SQRT(SUM(POW(dmid4, 2))), 0))             AS shp_rv_signature_ratio,

  -- ---- trends: where things were GOING, not where they sat --------------------
  SAFE_DIVIDE(COVAR_POP(depth_imb_1, t), NULLIF(VAR_POP(t), 0)) AS shp_imb_slope,
  SAFE_DIVIDE(COVAR_POP(spread, t), NULLIF(VAR_POP(t), 0))      AS shp_spread_slope,
  SAFE_DIVIDE(COVAR_POP(total_depth, t), NULLIF(VAR_POP(t), 0)) AS shp_depth_slope,
  SAFE_DIVIDE(COVAR_POP(mid, t), NULLIF(VAR_POP(t), 0))         AS shp_mid_slope,
  IF(IS_NAN(CORR(depth_imb_1, t)), NULL,
     CORR(depth_imb_1, t))                                     AS shp_imb_trend_corr,

  -- ---- half-window drift ------------------------------------------------------
  AVG(IF(rn > n_snaps / 2, depth_imb_1, NULL))
    - AVG(IF(rn <= n_snaps / 2, depth_imb_1, NULL))             AS shp_imb_drift,
  AVG(IF(rn > n_snaps / 2, spread, NULL))
    - AVG(IF(rn <= n_snaps / 2, spread, NULL))                  AS shp_spread_drift,
  AVG(IF(rn > n_snaps / 2, total_depth, NULL))
    - AVG(IF(rn <= n_snaps / 2, total_depth, NULL))             AS shp_depth_drift,

  -- ---- path extremes (not endpoint extremes) ----------------------------------
  -- Relative to the LAST snapshot, not an arbitrary one: ANY_VALUE would pick a
  -- non-deterministic row and make these two features unreproducible.
  SAFE_DIVIDE(MAX(mid) - MAX(IF(rn = n_snaps, mid, NULL)),
              NULLIF(MAX(IF(rn = n_snaps, mid, NULL)), 0))       AS shp_mid_runup_rel,
  SAFE_DIVIDE(MAX(IF(rn = n_snaps, mid, NULL)) - MIN(mid),
              NULLIF(MAX(IF(rn = n_snaps, mid, NULL)), 0))       AS shp_mid_drawdown_rel,

  -- ---- arrival process --------------------------------------------------------
  -- Regular quoting gives CV near 0; bursty, event-driven quoting gives CV above 1.
  SAFE_DIVIDE(STDDEV_POP(dt), NULLIF(ABS(AVG(dt)), 0))          AS shp_gap_cv,
  MAX(ABS(dt))                                                  AS shp_max_gap,

  COUNT(*)                                                      AS shp_n_snaps
FROM seq2
GROUP BY sample_id
"""


def run(split: str = "train", *, bq=None) -> dict:
    from google.cloud import bigquery

    from src.config import gcp_key_path

    bq = bq or bigquery.Client.from_service_account_json(str(gcp_key_path()))
    sql = build_sql(split)
    job = bq.query(sql)
    job.result()
    target = feature_table("shape", split)
    n = list(bq.query(f"SELECT COUNT(*) c FROM {target}").result())[0].c
    log.info("[shape/%s] %s rows, %.2f GB scanned", split, f"{n:,}",
             (job.total_bytes_processed or 0) / 1e9)
    return {"split": split, "rows": n, "bytes": job.total_bytes_processed}


def download(split: str = "train") -> object:
    """Pull the shape table down as Parquet, ordered by sample_id.

    Same ordering discipline as assemble.download(): bq.list_rows() returns rows in
    arbitrary order, and an artefact whose row order is meaningless is a trap for whoever
    reads it next.
    """
    from pathlib import Path

    import pyarrow.parquet as pq
    from google.cloud import bigquery

    from src.config import gcp_key_path, load_config

    bq = bigquery.Client.from_service_account_json(str(gcp_key_path()))
    cfg = load_config()
    tid = feature_table("shape", split).strip("`")
    dst = Path(cfg.paths.features) / f"shape_{split}.parquet"
    arrow = bq.query(f"SELECT * FROM `{tid}` ORDER BY sample_id").result().to_arrow(
        create_bqstorage_client=True)
    sid = arrow.column("sample_id").to_numpy()
    if not (sid[1:] > sid[:-1]).all():
        raise AssertionError(f"{tid}: rows are not strictly ordered by sample_id")
    pq.write_table(arrow, dst, compression="zstd")
    log.info("[shape/%s] downloaded %s: %.1f MB", split, dst.name,
             dst.stat().st_size / 1e6)
    return dst


def feature_names() -> list[str]:
    """Aliases emitted by the SQL, for tests and for the join step."""
    import re

    return re.findall(r"AS (shp_\w+)", build_sql("train"))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Build sequence-shape features")
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--download", action="store_true",
                    help="skip the build; just pull the table down")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if args.download:
        download(args.split)
    else:
        run(args.split)


if __name__ == "__main__":
    main()

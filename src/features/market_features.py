"""Market (L2 order-book snapshot) features - one row per sample.

221.8M snapshots / 1.26M samples = 176.3 snapshots on average.

NOTE - the market window is 600 SECONDS, unlike the 60 s of order/transaction.
1,257,631 of 1,257,637 samples exceed 60 s; the maximum observed is 599.911 s.
Snapshot cadence is 176.3/600 = one every ~3.4 s, so a 1-second window is NOT
used (it would be empty in ~71% of samples). Windows: {5,10,30,60,120,300,600} s.
Market has no row cap (max 212); the 999-row truncation of the other two tables
does not apply here.

HIGHEST-SIGNAL GROUP: the "last snapshot" values (seconds_before_predict -> 0),
i.e. the book state closest to the prediction instant.
"""
from __future__ import annotations

from src.features.common import cond, feature_table, safe_div, staged, wlabel, windows

NEWLINE_SEP = ",\n"

# Stage 1: CLEAN THE EMPTY-LEVEL SENTINEL.
# Measured over 221.8M rows: price = 0 ALWAYS coincides with volume = 0
#   bid_1 = 0 -> 133,781 rows | ask_1 = 0 -> 898,820 rows
# Real prices live in 0.90919 - 1.052, so 0 is not a price but a "this level is
# empty" sentinel. There are NO genuinely crossed books:
# bid_1 > 0 AND ask_1 > 0 AND bid_1 >= ask_1 returns 0 rows.
# Left uncleaned, mid/spread/microprice produce garbage (e.g. rel_spread = -2.0)
# and corrupt every mean, standard deviation and return built on them.
CLEAN = """
    sample_id,
    seconds_before_predict,
    NULLIF(bid_price_1, 0) AS bp1,
    NULLIF(ask_price_1, 0) AS ap1,
    NULLIF(bid_price_2, 0) AS bp2,
    NULLIF(ask_price_2, 0) AS ap2,
    CAST(bid_volume_1 AS FLOAT64) AS bv1,
    CAST(ask_volume_1 AS FLOAT64) AS av1,
    CAST(bid_volume_2 AS FLOAT64) AS bv2,
    CAST(ask_volume_2 AS FLOAT64) AS av2,
    transaction_avgprice,
    transaction_volume,
    transaction_count,
    IF(bid_price_1 = 0, 1, 0) AS is_empty_bid,
    IF(ask_price_1 = 0, 1, 0) AS is_empty_ask
"""

# Stage 2: microstructure quantities from the cleaned columns.
# If either side is empty, mid/spread/microprice become NULL - the correct behaviour.
ROW_DERIVED = """
    sample_id,
    seconds_before_predict,
    is_empty_bid,
    is_empty_ask,
    (ap1 + bp1) / 2                                          AS mid,
    ap1 - bp1                                                AS spread,
    SAFE_DIVIDE(ap1 - bp1, (ap1 + bp1) / 2)                  AS rel_spread,
    -- Microprice: mid weighted by the opposite side's volume. A better short-horizon
    -- fair-value estimate than the plain mid.
    SAFE_DIVIDE(ap1 * bv1 + bp1 * av1, NULLIF(bv1 + av1, 0)) AS microprice,
    -- Depth imbalance is only meaningful on a TWO-SIDED book. On a one-sided
    -- snapshot (e.g. ask_price_1 = 0 -> ask_volume_1 = 0) the formula degenerates
    -- to +-1: a valid but meaningless number that must not be mixed into the same
    -- distribution as two-sided values. The empty-side signal is kept separately
    -- as mkt_empty_bid_share / mkt_empty_ask_share.
    IF(ap1 IS NULL OR bp1 IS NULL, NULL,
       SAFE_DIVIDE(bv1 - av1, NULLIF(bv1 + av1, 0)))         AS depth_imb_1,
    IF(ap1 IS NULL OR bp1 IS NULL, NULL,
       SAFE_DIVIDE(bv1 + bv2 - av1 - av2,
                   NULLIF(bv1 + bv2 + av1 + av2, 0)))        AS depth_imb_12,
    bv1 + bv2 + av1 + av2                                    AS total_depth,
    -- Book slope: how far level 2 sits from level 1 (liquidity distribution).
    ap2 - ap1                                                AS ask_slope,
    bp1 - bp2                                                AS bid_slope,
    transaction_avgprice,
    transaction_volume,
    transaction_count
"""


def build_sql(split: str = "train") -> str:
    src = staged("market", split)
    base: list[str] = []

    for w in windows("market"):
        t = wlabel(w)
        c = cond(w)

        def avg(expr: str) -> str:
            return f"AVG(IF({c}, {expr}, NULL))"

        base += [
            f"    {avg('mid')} AS mkt_mid_mean_{t}",
            f"    {avg('spread')} AS mkt_spread_mean_{t}",
            f"    {avg('rel_spread')} AS mkt_rel_spread_mean_{t}",
            f"    {avg('depth_imb_1')} AS mkt_depth_imb1_mean_{t}",
            f"    {avg('depth_imb_12')} AS mkt_depth_imb12_mean_{t}",
            f"    {avg('microprice - mid')} AS mkt_micro_minus_mid_{t}",
            f"    {avg('ask_slope')} AS mkt_ask_slope_{t}",
            f"    {avg('bid_slope')} AS mkt_bid_slope_{t}",
            # Realised volatility proxy: standard deviation of the mid.
            f"    STDDEV(IF({c}, mid, NULL)) AS mkt_mid_std_{t}",
            f"    MAX(IF({c}, mid, NULL)) - MIN(IF({c}, mid, NULL)) AS mkt_mid_range_{t}",
            f"    STDDEV(IF({c}, depth_imb_1, NULL)) AS mkt_depth_imb1_std_{t}",
            f"    {avg('total_depth')} AS mkt_total_depth_mean_{t}",
            # Trade aggression: where the executed average price sits versus the mid.
            f"    {avg('SAFE_DIVIDE(transaction_avgprice - mid, NULLIF(spread, 0))')}"
            f" AS mkt_trade_aggression_{t}",
            f"    {safe_div(f'SUM(IF({c}, transaction_volume, 0))', str(w))}"
            f" AS mkt_trade_volume_rate_{t}",
            f"    {safe_div(f'SUM(IF({c}, transaction_count, 0))', str(w))}"
            f" AS mkt_trade_count_rate_{t}",
            f"    {safe_div(f'COUNTIF({c})', str(w))} AS mkt_snapshot_rate_{t}",
            # Mid at the start of the window - used for the return calculation.
            f"    ARRAY_AGG(IF({c}, mid, NULL) IGNORE NULLS"
            f" ORDER BY seconds_before_predict DESC LIMIT 1)[SAFE_OFFSET(0)]"
            f" AS mkt_mid_start_{t}",
        ]

    # LAST VALID SNAPSHOT - TAKEN FROM A SINGLE ROW.
    #
    # A separate ARRAY_AGG(... IGNORE NULLS) per field is NOT used: each field
    # would pick its own first non-null value and the fields would come from
    # DIFFERENT snapshots. Measured: that bug made mkt_depth_imb1_last deviate by
    # 1.994 from an independent recomputation - the full width of [-1, 1].
    # Fix: take the valid (two-sided) snapshot closest to the prediction instant
    # as one STRUCT and read every *_last field from it.
    LAST_FIELDS = [
        ("mid", "mkt_mid_last"),
        ("spread", "mkt_spread_last"),
        ("rel_spread", "mkt_rel_spread_last"),
        ("microprice", "mkt_microprice_last"),
        ("depth_imb_1", "mkt_depth_imb1_last"),
        ("depth_imb_12", "mkt_depth_imb12_last"),
        ("total_depth", "mkt_total_depth_last"),
        ("ask_slope", "mkt_ask_slope_last"),
        ("bid_slope", "mkt_bid_slope_last"),
        ("seconds_before_predict", "mkt_last_valid_gap"),
    ]
    struct_fields = ", ".join(src_col for src_col, _ in LAST_FIELDS)
    base += [
        "    ARRAY_AGG(IF(mid IS NOT NULL, STRUCT("
        f"{struct_fields}), NULL)"
        " IGNORE NULLS ORDER BY seconds_before_predict ASC LIMIT 1)[SAFE_OFFSET(0)]"
        " AS last_snap",
        "    AVG(is_empty_bid) AS mkt_empty_bid_share",
        "    AVG(is_empty_ask) AS mkt_empty_ask_share",
        "    COUNTIF(mid IS NULL) / COUNT(*) AS mkt_null_mid_share",
        "    AVG(spread) AS mkt_spread_mean_clean",
        "    MIN(spread) AS mkt_spread_min",
        "    MIN(seconds_before_predict) AS mkt_last_seconds_gap",
        "    COUNT(*) AS mkt_n_snapshots",
    ]

    # Returns and last-snapshot derivatives go in an outer query so they can
    # reference the aliases above.
    ws = windows("market")
    long_label = wlabel(ws[-1])
    derived = [
        f"  {safe_div('mkt_mid_last', 'mkt_mid_start_' + wlabel(w))} - 1"
        f" AS mkt_mid_return_{wlabel(w)}"
        for w in ws
    ] + [
        "  mkt_microprice_last - mkt_mid_last AS mkt_micro_minus_mid_last",
        f"  {safe_div('mkt_microprice_last - mkt_mid_last', 'mkt_spread_last')}"
        " AS mkt_micro_edge_norm_last",
        # Instantaneous spread versus its window average: a liquidity-stress gauge.
        f"  {safe_div('mkt_spread_last', 'mkt_spread_mean_' + long_label)}"
        " AS mkt_spread_last_vs_mean",
        f"  {safe_div('mkt_total_depth_last', 'mkt_total_depth_mean_' + long_label)}"
        " AS mkt_depth_last_vs_mean",
    ] + [
        f"  {safe_div('mkt_snapshot_rate_' + wlabel(s), 'mkt_snapshot_rate_' + long_label)}"
        f" AS mkt_snapshot_accel_{wlabel(s)}_vs_{long_label}"
        for s in ws[:-1]
    ] + [
        f"  mkt_depth_imb1_mean_{wlabel(s)} - mkt_depth_imb1_mean_{long_label}"
        f" AS mkt_depth_imb1_delta_{wlabel(s)}_vs_{long_label}"
        for s in ws[:-1]
    ]

    base_sql = NEWLINE_SEP.join(base)
    derived_sql = NEWLINE_SEP.join(derived)
    flat_sql = NEWLINE_SEP.join(
        f"  last_snap.{src_col} AS {alias}" for src_col, alias in LAST_FIELDS
    )
    target = feature_table("market", split)
    return (
        f"CREATE OR REPLACE TABLE {target}\n"
        "CLUSTER BY sample_id AS\n"
        "WITH clean AS (\n"
        f"  SELECT{CLEAN}"
        f"  FROM {src}\n"
        "),\n"
        "rows_ AS (\n"
        f"  SELECT{ROW_DERIVED}"
        "  FROM clean\n"
        "),\n"
        "agg AS (\n"
        "  SELECT\n"
        "    sample_id,\n"
        f"{base_sql}\n"
        "  FROM rows_\n"
        "  GROUP BY sample_id\n"
        "),\n"
        # Flatten the last_snap STRUCT so the derived expressions
        # (mkt_mid_return_*, mkt_micro_edge_norm_last, ...) can use the aliases.
        "flat AS (\n"
        "  SELECT\n"
        "    agg.* EXCEPT (last_snap),\n"
        f"{flat_sql}\n"
        "  FROM agg\n"
        ")\n"
        "SELECT\n"
        "  flat.*,\n"
        f"{derived_sql}\n"
        "FROM flat\n"
    )

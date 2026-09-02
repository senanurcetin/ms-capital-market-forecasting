"""Transaction (executed trade) features - one row per sample.

104.0M trades / 1.26M samples = 82.7 trades on average, over a 60 s window.

The `side` column is the aggressor side. Verified against the prevailing market
mid over 1.71M trades: side=0 trades sit 87.5% above mid (mean +5.26 bps) -> BUY;
side=1 sit 88.7% below mid (mean -5.27 bps) -> SELL.
"""
from __future__ import annotations

from src.features.common import (
    cond, feature_table, imbalance, row_cap, safe_div, staged, wlabel, windows,
)

NEWLINE_SEP = ",\n"

BUY, SELL = 0, 1


def build_sql(split: str = "train") -> str:
    src = staged("transaction", split)
    base: list[str] = []

    for w in windows("transaction"):
        t = wlabel(w)
        c = cond(w)
        buy_v = f"SUM(IF({c} AND side = {BUY}, volume, 0))"
        sell_v = f"SUM(IF({c} AND side = {SELL}, volume, 0))"
        buy_n = f"COUNTIF({c} AND side = {BUY})"
        sell_n = f"COUNTIF({c} AND side = {SELL})"
        n = f"COUNTIF({c})"
        vol = f"SUM(IF({c}, volume, 0))"
        notional = f"SUM(IF({c}, price * volume, 0))"

        base += [
            # Rates, not raw counts - robust to the train/test density difference.
            f"    {safe_div(n, str(w))} AS txn_intensity_{t}",
            f"    {safe_div(vol, str(w))} AS txn_volume_rate_{t}",
            # Imbalances in [-1, 1], scale-free.
            f"    {imbalance(buy_v, sell_v)} AS txn_volume_imbalance_{t}",
            f"    {imbalance(buy_n, sell_n)} AS txn_count_imbalance_{t}",
            # VWAP and average trade size.
            f"    {safe_div(notional, vol)} AS txn_vwap_{t}",
            f"    {safe_div(vol, n)} AS txn_avg_size_{t}",
            # Price dispersion - a realised volatility proxy.
            f"    STDDEV(IF({c}, price, NULL)) AS txn_price_std_{t}",
            f"    MAX(IF({c}, price, NULL)) - MIN(IF({c}, price, NULL)) AS txn_price_range_{t}",
        ]

    # seconds_before_predict is sorted descending, so MIN(seconds) is the trade
    # CLOSEST to the prediction instant.
    last_price = "ARRAY_AGG(price ORDER BY seconds_before_predict ASC LIMIT 1)[OFFSET(0)]"
    first_price = "ARRAY_AGG(price ORDER BY seconds_before_predict DESC LIMIT 1)[OFFSET(0)]"
    base += [
        f"    {last_price} AS txn_last_price",
        f"    {first_price} AS txn_first_price",
        f"    {safe_div(last_price, first_price)} - 1 AS txn_window_return",
        "    MIN(seconds_before_predict) AS txn_last_seconds_gap",
        "    COUNT(*) AS txn_n_total",
        # Truncation signal: samples cap at exactly row_cap rows, so a capped
        # sample effectively covers less than the full 60 s. Keep both the flag
        # and the actual covered span.
        f"    IF(COUNT(*) >= {row_cap()}, 1, 0) AS txn_is_truncated",
        "    MAX(seconds_before_predict) AS txn_window_covered",
        f"    {safe_div('SUM(price * volume)', 'SUM(volume)')} AS txn_vwap_total",
        f"    {safe_div('SUM(IF(volume >= 10000, volume, 0))', 'SUM(volume)')}"
        " AS txn_large_volume_share",
    ]

    # Momentum / acceleration: short window over long window. Aliases cannot be
    # referenced in the same SELECT, so these go in an outer query.
    ws = windows("transaction")
    long_label = wlabel(ws[-1])
    accel = [
        f"  {safe_div('txn_intensity_' + wlabel(s), 'txn_intensity_' + long_label)}"
        f" AS txn_intensity_accel_{wlabel(s)}_vs_{long_label}"
        for s in ws[:-1]
    ]

    base_sql = NEWLINE_SEP.join(base)
    accel_sql = NEWLINE_SEP.join(accel)
    target = feature_table("transaction", split)
    return (
        f"CREATE OR REPLACE TABLE {target}\n"
        "CLUSTER BY sample_id AS\n"
        "WITH agg AS (\n"
        "  SELECT\n"
        "    sample_id,\n"
        f"{base_sql}\n"
        f"  FROM {src}\n"
        "  GROUP BY sample_id\n"
        ")\n"
        "SELECT\n"
        "  agg.*,\n"
        f"{accel_sql}\n"
        "FROM agg\n"
    )

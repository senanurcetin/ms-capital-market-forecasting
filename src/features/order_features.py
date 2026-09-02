"""Order (emir akisi) feature'lari - sample basina tek satir.

170.1M emir olayi / 1.26M sample = ortalama 135.2 olay.

KODLAMA (EDA ile ampirik cozuldu, configs/config.yaml -> encoding):
    side   0 = BID (alis)   1 = ASK (satis)
    action 0 = NEW          1 = CANCEL

DIKKAT: test'te sample basina emir yogunlugu %36 daha yuksek (135.2 -> 184.4).
Bu yuzden ham sayimlar degil, ORAN ve YOGUNLUK formlari uretilir.
"""
from __future__ import annotations

from src.features.common import (
    cond, feature_table, imbalance, row_cap, safe_div, staged, wlabel, windows,
)

NEWLINE_SEP = ",\n"

BID, ASK = 0, 1
NEW, CANCEL = 0, 1


def build_sql(split: str = "train") -> str:
    src = staged("order", split)
    base: list[str] = []

    for w in windows("order"):
        t = wlabel(w)
        c = cond(w)

        def cnt(side: int, action: int) -> str:
            return f"COUNTIF({c} AND side = {side} AND order_action = {action})"

        def vol(side: int, action: int) -> str:
            return f"SUM(IF({c} AND side = {side} AND order_action = {action}, volume, 0))"

        new_bid, new_ask = cnt(BID, NEW), cnt(ASK, NEW)
        cxl_bid, cxl_ask = cnt(BID, CANCEL), cnt(ASK, CANCEL)
        nv_bid, nv_ask = vol(BID, NEW), vol(ASK, NEW)
        cv_bid, cv_ask = vol(BID, CANCEL), vol(ASK, CANCEL)
        n_all = f"COUNTIF({c})"

        base += [
            # Gelme / iptal hizlari (olay/saniye)
            f"    {safe_div(f'{new_bid} + {new_ask}', str(w))} AS ord_new_rate_{t}",
            f"    {safe_div(f'{cxl_bid} + {cxl_ask}', str(w))} AS ord_cancel_rate_{t}",
            # Iptal / yeni orani: likidite saglayicilarinin geri cekilme sinyali
            f"    {safe_div(f'{cxl_bid} + {cxl_ask}', f'{new_bid} + {new_ask}')}"
            f" AS ord_cancel_new_ratio_{t}",
            # Order Flow Imbalance: net eklenen derinlik (bid) - (ask)
            f"    {safe_div(f'({nv_bid} - {cv_bid}) - ({nv_ask} - {cv_ask})', f'{nv_bid} + {cv_bid} + {nv_ask} + {cv_ask}')}"
            f" AS ord_ofi_{t}",
            # Yeni emir dengesizligi (sayim ve hacim)
            f"    {imbalance(new_bid, new_ask)} AS ord_new_count_imbalance_{t}",
            f"    {imbalance(nv_bid, nv_ask)} AS ord_new_volume_imbalance_{t}",
            # Iptal dengesizligi: tek tarafli cekilme yon sinyali tasir
            f"    {imbalance(cxl_bid, cxl_ask)} AS ord_cancel_count_imbalance_{t}",
            f"    {imbalance(cv_bid, cv_ask)} AS ord_cancel_volume_imbalance_{t}",
            # Emir buyuklugu
            f"    {safe_div(f'{nv_bid} + {nv_ask}', f'{new_bid} + {new_ask}')}"
            f" AS ord_avg_new_size_{t}",
            f"    {safe_div(f'{cv_bid} + {cv_ask}', f'{cxl_bid} + {cxl_ask}')}"
            f" AS ord_avg_cancel_size_{t}",
            f"    {safe_div(n_all, str(w))} AS ord_event_rate_{t}",
            # Fiyat konumu: emirlerin agirlikli ortalama fiyati (mid ~ 1.0'a gore)
            f"    {safe_div(f'SUM(IF({c} AND side = {BID}, price * volume, 0))', f'SUM(IF({c} AND side = {BID}, volume, 0))')}"
            f" AS ord_bid_vwap_{t}",
            f"    {safe_div(f'SUM(IF({c} AND side = {ASK}, price * volume, 0))', f'SUM(IF({c} AND side = {ASK}, volume, 0))')}"
            f" AS ord_ask_vwap_{t}",
        ]

    # Ustel zaman agirligi: yakin olaylara daha fazla agirlik (lambda = 1/10 sn)
    for lam in (0.1, 0.5):
        tag = f"{lam:g}".replace(".", "p")
        wb = f"SUM(IF(side = {BID} AND order_action = {NEW}, volume * EXP(-{lam} * seconds_before_predict), 0))"
        wa = f"SUM(IF(side = {ASK} AND order_action = {NEW}, volume * EXP(-{lam} * seconds_before_predict), 0))"
        base.append(f"    {imbalance(wb, wa)} AS ord_decay_imbalance_lam{tag}")

    base += [
        "    COUNT(*) AS ord_n_total",
        # Kirpma sinyali: sample basina TAM row_cap() satirda tavan var. Kirpilan
        # sample'larda pencere fiilen 60 sn'den kisadir -> gercek kapsanan sureyi de tut.
        f"    IF(COUNT(*) >= {row_cap()}, 1, 0) AS ord_is_truncated",
        "    MAX(seconds_before_predict) AS ord_window_covered",
        "    MIN(seconds_before_predict) AS ord_last_seconds_gap",
        f"    {safe_div(f'COUNTIF(order_action = {CANCEL})', 'COUNT(*)')} AS ord_cancel_share_total",
        "    STDDEV(volume) AS ord_volume_std",
        f"    {safe_div('MAX(volume)', 'AVG(volume)')} AS ord_max_to_avg_size",
    ]

    ws = windows("order")
    long_label = wlabel(ws[-1])
    accel = [
        f"  {safe_div('ord_event_rate_' + wlabel(s), 'ord_event_rate_' + long_label)}"
        f" AS ord_event_accel_{wlabel(s)}_vs_{long_label}"
        for s in ws[:-1]
    ] + [
        f"  ord_ofi_{wlabel(s)} - ord_ofi_{long_label}"
        f" AS ord_ofi_delta_{wlabel(s)}_vs_{long_label}"
        for s in ws[:-1]
    ]

    base_sql = NEWLINE_SEP.join(base)
    accel_sql = NEWLINE_SEP.join(accel)
    target = feature_table("order", split)
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

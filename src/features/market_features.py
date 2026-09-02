"""Market (L2 order book snapshot) feature'lari - sample basina tek satir.

221.8M snapshot / 1.26M sample = ortalama 176.3 snapshot.

DIKKAT - market penceresi 600 SANIYE (10 dk), order/transaction'in 60 sn'sinden farkli.
1.26M sample'in 1,257,631'i 60 sn'yi asiyor; max 599.911 sn. Snapshot sikligi
176.3/600 = ~3.4 saniyede bir, bu yuzden 1 sn'lik pencere KULLANILMAZ (orneklerin
~%71'inde bos kalirdi). Pencereler: {5, 10, 30, 60, 120, 300, 600} sn.
Market'te satir tavani yok (max 212), order/transaction'daki 999 kirpmasi burada yok.

EN YUKSEK SINYAL BEKLENEN GRUP: "son snapshot" degerleri
(seconds_before_predict -> 0), yani tahmin anina en yakin defter durumu.
"""
from __future__ import annotations

from src.features.common import cond, feature_table, safe_div, staged, wlabel, windows

NEWLINE_SEP = ",\n"

# Satir basina turetilen buyuklukler. Fiyatlar sample bazinda ~1.0'a normalize.
# 1. asama: BOS SEVIYE SENTINEL'INI TEMIZLE.
# Olculdu (221.8M satir): price = 0 HER ZAMAN volume = 0 ile birlikte gelir
#   bid_1 = 0 -> 133,781 satir | ask_1 = 0 -> 898,820 satir
# Gercek fiyatlar 0.90919 - 1.052 araliginda; 0 bir fiyat degil, "bu seviye bos" sentinel'i.
# GERCEK CAPRAZ DEFTER YOK: bid_1 > 0 AND ask_1 > 0 AND bid_1 >= ask_1 -> 0 satir.
# Sentinel temizlenmezse mid/spread/microprice cop uretir (ornegin rel_spread = -2.0)
# ve tum ortalamalari, std'leri, getirileri bozar.
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

# 2. asama: temizlenmis kolonlardan mikroyapi buyuklukleri.
# Herhangi bir taraf bos ise mid/spread/microprice NULL olur - dogru davranis.
ROW_DERIVED = """
    sample_id,
    seconds_before_predict,
    is_empty_bid,
    is_empty_ask,
    (ap1 + bp1) / 2                                          AS mid,
    ap1 - bp1                                                AS spread,
    SAFE_DIVIDE(ap1 - bp1, (ap1 + bp1) / 2)                  AS rel_spread,
    -- Mikro fiyat: karsit hacimle agirliklandirilmis mid; kisa vadede mid'den
    -- daha iyi bir "gercek deger" tahmincisidir.
    SAFE_DIVIDE(ap1 * bv1 + bp1 * av1, NULLIF(bv1 + av1, 0)) AS microprice,
    -- Derinlik dengesizligi YALNIZ cift tarafli defterde anlamli. Tek tarafli
    -- snapshot'ta (ornegin ask_price_1 = 0 -> ask_volume_1 = 0) formul otomatik
    -- olarak +-1 verir; bu gecerli ama DEJENERE bir sayidir ve iki tarafli
    -- degerlerle ayni dagilima karistirilmamalidir. Bos taraf sinyali zaten
    -- mkt_empty_bid_share / mkt_empty_ask_share ile ayrica tutuluyor.
    IF(ap1 IS NULL OR bp1 IS NULL, NULL,
       SAFE_DIVIDE(bv1 - av1, NULLIF(bv1 + av1, 0)))         AS depth_imb_1,
    IF(ap1 IS NULL OR bp1 IS NULL, NULL,
       SAFE_DIVIDE(bv1 + bv2 - av1 - av2,
                   NULLIF(bv1 + bv2 + av1 + av2, 0)))        AS depth_imb_12,
    bv1 + bv2 + av1 + av2                                    AS total_depth,
    -- Defter egimi: 2. seviyenin 1. seviyeden uzakligi (likidite dagilimi)
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
            # Gerceklesmis volatilite vekili: mid'in standart sapmasi
            f"    STDDEV(IF({c}, mid, NULL)) AS mkt_mid_std_{t}",
            f"    MAX(IF({c}, mid, NULL)) - MIN(IF({c}, mid, NULL)) AS mkt_mid_range_{t}",
            f"    STDDEV(IF({c}, depth_imb_1, NULL)) AS mkt_depth_imb1_std_{t}",
            # Derinlik: seviye olarak degil, sample icinde normalize edilerek
            f"    {avg('total_depth')} AS mkt_total_depth_mean_{t}",
            # Islem agresyonu: gerceklesen ortalama fiyat mid'in neresinde?
            f"    {avg('SAFE_DIVIDE(transaction_avgprice - mid, NULLIF(spread, 0))')}"
            f" AS mkt_trade_aggression_{t}",
            f"    {safe_div(f'SUM(IF({c}, transaction_volume, 0))', str(w))}"
            f" AS mkt_trade_volume_rate_{t}",
            f"    {safe_div(f'SUM(IF({c}, transaction_count, 0))', str(w))}"
            f" AS mkt_trade_count_rate_{t}",
            f"    {safe_div(f'COUNTIF({c})', str(w))} AS mkt_snapshot_rate_{t}",
            # Pencere basindaki mid: getiri hesabi icin
            f"    ARRAY_AGG(IF({c}, mid, NULL) IGNORE NULLS"
            f" ORDER BY seconds_before_predict DESC LIMIT 1)[SAFE_OFFSET(0)]"
            f" AS mkt_mid_start_{t}",
        ]

    # SON GECERLI SNAPSHOT - TEK BIR SATIRDAN.
    #
    # Alan basina ayri ARRAY_AGG(... IGNORE NULLS) KULLANILMAZ: her alan kendi
    # ilk non-null degerini secerdi ve alanlar FARKLI snapshot'lardan gelirdi.
    # Olculdu: bu hata mkt_depth_imb1_last'te bagimsiz yeniden hesaba gore
    # 1.994 (yani [-1,1] araliginin tamami kadar) sapma uretiyordu.
    # Cozum: gecerli (cift tarafli) snapshot'lar arasindan tahmin anina en
    # yakini bir STRUCT olarak alinir, tum *_last alanlari ondan okunur.
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

    # Getiriler ve son-snapshot turevleri ust sorguda (alias kullanimi icin)
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
        # Anlik spread'in pencere ortalamasina orani: likidite stresi gostergesi
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
        # last_snap STRUCT'ini duz kolonlara ac: turetilmis ifadeler
        # (mkt_mid_return_*, mkt_micro_edge_norm_last, ...) bu alias'lari
        # kullanabilsin diye ayri bir katman gerekiyor.
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

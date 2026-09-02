"""Feature SQL ureticileri - yapisal garantiler.

Bu testler SQL'i calistirmadan, uretilen metin uzerinden dogrulama yapar.
Amac: BigQuery'ye gitmeden look-ahead, pencere ve isimlendirme kurallarini
her degisiklikte otomatik yakalamak.
"""
import itertools
import re

import pytest

from src.config import load_config
from src.features import (
    assemble, common, market_features, order_features, transaction_features,
)

MODULES = {
    "market": (market_features, "mkt_"),
    "order": (order_features, "ord_"),
    "transaction": (transaction_features, "txn_"),
}


def aliases(sql: str, prefix: str) -> set[str]:
    return {a for a in re.findall(r"\sAS\s+([a-z][a-z0-9_]*)", sql) if a.startswith(prefix)}


@pytest.mark.parametrize("table", list(MODULES))
def test_no_future_looking_comparison(table):
    """Pencere filtreleri YALNIZ 'seconds_before_predict <= W' olmali.

    seconds_before_predict tahmin anina olan GERIYE uzakliktir; '>=' veya '>'
    kullanmak 'daha eski veri' demek olur ve pencere semantigini bozar.
    Gelecege bakmak veri yapisi geregi zaten imkansiz (deger >= 0), ama bu
    test yanlis yonlu bir filtreyi de yakalar.
    """
    sql = MODULES[table][0].build_sql("train")
    bad = re.findall(r"seconds_before_predict\s*(>=|>)\s*[0-9]", sql)
    assert not bad, f"{table}: ters yonlu pencere filtresi {bad}"
    assert "seconds_before_predict <=" in sql


@pytest.mark.parametrize("table", list(MODULES))
def test_windows_match_config(table):
    """Uretilen pencere etiketleri config'deki listeyle birebir ortusmeli."""
    sql = MODULES[table][0].build_sql("train")
    expected = {common.wlabel(w) for w in common.windows(table)}
    found = set(re.findall(r"_(\d+p?\d*s)\b", sql))
    assert expected <= found, f"{table}: eksik pencere {expected - found}"
    assert found <= expected, f"{table}: config'de olmayan pencere {found - expected}"


def test_market_uses_600s_window_and_no_1s():
    """Market penceresi 600 sn; 1 sn snapshot sikligi (~3.4 sn) nedeniyle yok."""
    sql = market_features.build_sql("train")
    assert "mkt_mid_mean_600s" in sql
    assert "mkt_mid_mean_1s" not in sql
    assert load_config().window.seconds["market"] == 600.0


@pytest.mark.parametrize("table", list(MODULES))
def test_reads_only_its_own_staging_table(table):
    """Bir feature modulu baska bir tabloya dokunmamali (granularite karisimi)."""
    sql = MODULES[table][0].build_sql("train")
    others = [t for t in MODULES if t != table]
    for other in others:
        assert f".{other}_train`" not in sql, f"{table} SQL'i {other} tablosunu okuyor"


@pytest.mark.parametrize("table", list(MODULES))
def test_no_target_or_month_leak(table):
    """Feature SQL'i target'a veya month'a asla dokunmamali."""
    sql = MODULES[table][0].build_sql("train")
    assert "target" not in sql
    assert not re.search(r"\bmonth\b", sql), f"{table}: month feature katmaninda kullanilmis"


@pytest.mark.parametrize("table", list(MODULES))
def test_all_aliases_prefixed_and_unique(table):
    mod, prefix = MODULES[table]
    sql = mod.build_sql("train")
    found = aliases(sql, prefix)
    assert found, f"{table}: hic feature uretilmemis"
    all_aliases = re.findall(r"\sAS\s+([a-z][a-z0-9_]*)", sql)
    feature_aliases = [a for a in all_aliases if a.startswith(("mkt_", "ord_", "txn_"))]
    assert len(feature_aliases) == len(set(feature_aliases)), f"{table}: tekrarli alias"


def test_no_alias_collision_between_modules():
    sets = {t: aliases(m.build_sql("train"), p) for t, (m, p) in MODULES.items()}
    for a, b in itertools.combinations(sets, 2):
        assert not sets[a] & sets[b], f"{a}/{b} cakismasi: {sets[a] & sets[b]}"


def test_market_last_fields_come_from_single_snapshot():
    """*_last alanlari TEK bir STRUCT'tan okunmali.

    Alan basina ARRAY_AGG(... IGNORE NULLS) kullanmak alanlarin FARKLI
    snapshot'lardan gelmesine yol aciyordu; olculen sapma 1.994 idi.
    """
    sql = market_features.build_sql("train")
    assert "last_snap" in sql and "EXCEPT (last_snap)" in sql

    # Her *_last alani ya DOGRUDAN STRUCT'tan okunur, ya da yalnizca diger
    # *_last alanlarindan turetilir. Ikinci durumda da tek snapshot korunur.
    direct = set(re.findall(r"last_snap\.\w+ AS (\w+)", sql))
    assert direct, "STRUCT'tan hic alan okunmamis"

    for alias in [a for a in aliases(sql, "mkt_") if a.endswith("_last")]:
        if alias in direct:
            continue
        expr = re.search(rf"^(.*?)\s+AS {alias}\b", sql, re.MULTILINE)
        assert expr, f"{alias} icin ifade bulunamadi"
        referenced = {r for r in re.findall(r"\bmkt_\w+", expr.group(1))}
        assert referenced, f"{alias} turetilmis gorunmuyor ama STRUCT'tan da gelmiyor"
        assert referenced <= direct, (
            f"{alias}, tek snapshot disindaki alanlardan turetilmis: {referenced - direct}"
        )

    # Alan basina ayri ARRAY_AGG kalmadigini da dogrula (eski hatanin imzasi)
    per_field_aggs = re.findall(r"ARRAY_AGG\((?!IF\(mid IS NOT NULL)[^)]*\)\s*IGNORE NULLS"
                                r"\s*ORDER BY seconds_before_predict ASC", sql)
    assert not per_field_aggs, f"alan basina ARRAY_AGG kalmis: {per_field_aggs}"


def test_depth_imbalance_guarded_on_two_sided_book():
    sql = market_features.build_sql("train")
    assert "IF(ap1 IS NULL OR bp1 IS NULL, NULL," in sql


def test_truncation_features_present_for_capped_tables():
    """order/transaction sample basina 999 satirda taniyor -> kirpma sinyali."""
    cfg = load_config()
    for table in cfg.truncation["tables"]:
        sql = MODULES[table][0].build_sql("train")
        prefix = MODULES[table][1]
        assert f"{prefix}is_truncated" in sql
        assert f"{prefix}window_covered" in sql
        assert str(cfg.truncation["row_cap"]) in sql


def test_assemble_includes_label_for_train_only():
    train_sql = assemble.assemble_sql("train")
    test_sql = assemble.assemble_sql("test")
    assert "lbl.month" in train_sql and "lbl.target" in train_sql
    assert "lbl." not in test_sql, "test'te month/target YOK - yarisma vermiyor"
    assert "PARTITION BY RANGE_BUCKET(month" in train_sql
    assert "PARTITION BY" not in test_sql


def test_assemble_joins_all_three_blocks():
    sql = assemble.assemble_sql("train")
    for block in ("market_train", "order_train", "transaction_train"):
        assert block in sql
    assert sql.count("EXCEPT (sample_id)") == 3

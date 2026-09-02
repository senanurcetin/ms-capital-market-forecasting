"""Feature SQL uretimi icin ortak yardimcilar.

TASARIM KURALLARI (Bolum A bulgularindan):
  1. Sembol kolonu YOK -> tum feature'lar sample ICINDE hesaplanir.
     "Rolling" burada pencere icindeki IC ICE zaman dilimleridir
     (filtre: seconds_before_predict <= W).
  1b. PENCERE UZUNLUKLARI TABLOYA GORE FARKLI (olculdu):
     market 600 sn (~3.4 sn'de bir snapshot) -> W in {5,10,30,60,120,300,600}
     order / transaction 60 sn              -> W in {1,5,10,30,60}
     Market'e 60 sn'lik pencere uygulamak gecmisin %90'ini atmak olurdu.
  2. Look-ahead yapisal olarak imkansiz (her satir seconds_before_predict >= 0),
     ama pencere filtreleri yine de yalniz gecmise bakar.
  3. Test'te sample basina order yogunlugu %36 daha yuksek (train 135.2 -> test 184.4).
     Bu yuzden HAM SAYIM/HACIM yerine ORAN, YOGUNLUK ve NORMALIZE formlar tercih edilir.
"""
from __future__ import annotations

from src.config import load_config

# Kisa etiketler: 1.0 -> "1s", 0.5 -> "0p5s"
def wlabel(w: float) -> str:
    return f"{w:g}".replace(".", "p") + "s"


def windows(table: str) -> list[float]:
    """Tabloya ozgu ic ice pencere listesi (market 600 sn, digerleri 60 sn)."""
    return list(load_config().window.nested[table])


def full_window(table: str) -> float:
    return float(load_config().window.seconds[table])


def row_cap() -> int:
    """order/transaction sample basina TAM 999 satirda tavan yapiyor."""
    return int(load_config().truncation["row_cap"])


def cond(w: float) -> str:
    """Pencere filtresi. seconds_before_predict tahmin anina olan uzakliktir,
    dolayisiyla <= W demek 'son W saniye' demektir."""
    return f"seconds_before_predict <= {w:g}"


def safe_div(num: str, den: str, default: str = "NULL") -> str:
    """Sifira bolmeyi engelleyen guvenli bolme."""
    return f"SAFE_DIVIDE({num}, NULLIF({den}, 0))"


def imbalance(a: str, b: str) -> str:
    """(a - b) / (a + b) -> [-1, 1]. Olcek-bagimsiz oldugu icin train/test
    yogunluk farkindan etkilenmez."""
    return safe_div(f"({a}) - ({b})", f"({a}) + ({b})")


def staged(table: str, split: str) -> str:
    cfg = load_config()
    return f"`{cfg.bigquery.project}.{cfg.bigquery.datasets.staging}.{table}_{split}`"


def feature_table(name: str, split: str) -> str:
    cfg = load_config()
    return f"`{cfg.bigquery.project}.{cfg.bigquery.datasets.features}.{name}_{split}`"

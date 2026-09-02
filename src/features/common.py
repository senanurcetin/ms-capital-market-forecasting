"""Shared helpers for generating feature SQL.

DESIGN RULES (derived from the dataset analysis):
  1. There is NO symbol/instrument column, so every feature is computed WITHIN a
     sample. "Rolling" here means nested time slices inside the sample's own
     lookback window (filter: seconds_before_predict <= W).
  1b. WINDOW LENGTHS DIFFER PER TABLE (measured):
     market 600 s (~one snapshot every 3.4 s) -> W in {5,10,30,60,120,300,600}
     order / transaction 60 s                 -> W in {1,5,10,30,60}
     Applying a 60 s window to market would discard 90% of its history.
  2. Look-ahead is structurally impossible (seconds_before_predict >= 0), but the
     window filters still only ever look backwards.
  3. Test has ~36% more order events per sample than train (135.2 -> 184.4), so
     RATIOS, RATES and NORMALISED forms are preferred over raw counts and sums.
"""
from __future__ import annotations

from src.config import load_config


def wlabel(w: float) -> str:
    """Short window label: 1.0 -> "1s", 0.5 -> "0p5s"."""
    return f"{w:g}".replace(".", "p") + "s"


def windows(table: str) -> list[float]:
    """Nested window list for a table (market 600 s, the others 60 s)."""
    return list(load_config().window.nested[table])


def full_window(table: str) -> float:
    return float(load_config().window.seconds[table])


def row_cap() -> int:
    """order/transaction cap at exactly 999 rows per sample."""
    return int(load_config().truncation["row_cap"])


def cond(w: float) -> str:
    """Window filter. seconds_before_predict is the distance back from the
    prediction instant, so <= W means "within the last W seconds"."""
    return f"seconds_before_predict <= {w:g}"


def safe_div(num: str, den: str, default: str = "NULL") -> str:
    """Division that cannot blow up on a zero denominator."""
    return f"SAFE_DIVIDE({num}, NULLIF({den}, 0))"


def imbalance(a: str, b: str) -> str:
    """(a - b) / (a + b) -> [-1, 1]. Scale-free, so it is unaffected by the
    train/test difference in event density."""
    return safe_div(f"({a}) - ({b})", f"({a}) + ({b})")


def staged(table: str, split: str) -> str:
    cfg = load_config()
    return f"`{cfg.bigquery.project}.{cfg.bigquery.datasets.staging}.{table}_{split}`"


def feature_table(name: str, split: str) -> str:
    cfg = load_config()
    return f"`{cfg.bigquery.project}.{cfg.bigquery.datasets.features}.{name}_{split}`"

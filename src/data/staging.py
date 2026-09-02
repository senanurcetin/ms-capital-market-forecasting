"""raw (column groups) -> staging (canonical, partitioned and clustered).

Train tables are partitioned by month (71 partitions) so a walk-forward fold only
scans the months it needs, which keeps both scan cost and runtime linear in the
fold size. Test has no month column, so it is partitioned by sample_id buckets.
"""
from __future__ import annotations

import logging

from google.cloud import bigquery

from src.config import REPO_ROOT, load_config
from src.data.bq_loader import client

log = logging.getLogger(__name__)


def _table_exists(bq: bigquery.Client, table_id: str) -> bool:
    from google.cloud.exceptions import NotFound

    try:
        bq.get_table(table_id)
        return True
    except NotFound:
        return False


SQL_DIR = REPO_ROOT / "sql"

# Train: partition by month, which arrives via the label join.
_TRAIN_PARTITION = "PARTITION BY RANGE_BUCKET(month, GENERATE_ARRAY(0, 72, 1))"
# Test: no month -> sample_id buckets (647,896 samples / 5000 = 130 partitions).
_TEST_PARTITION = (
    "PARTITION BY RANGE_BUCKET(sample_id, GENERATE_ARRAY(0, 650000, 5000))"
)


def _render(template: str, split: str) -> str:
    cfg = load_config()
    is_train = split == "train"
    return template.format(
        project=cfg.bigquery.project,
        raw=cfg.bigquery.datasets.raw,
        staging=cfg.bigquery.datasets.staging,
        split=split,
        partition_clause=_TRAIN_PARTITION if is_train else _TEST_PARTITION,
        month_select="lbl.month," if is_train else "",
        # USING (sample_id) is ambiguous for market: g1/g2/g3 all carry that column.
        # Hence an explicit ON clause bound to the g1 alias.
        month_join=(
            f"JOIN `{cfg.bigquery.project}.{cfg.bigquery.datasets.staging}.label` AS lbl "
            "ON lbl.sample_id = g1.sample_id"
            if is_train
            else ""
        ),
    )


def run_sql_file(name: str, split: str, *, bq: bigquery.Client | None = None) -> dict:
    bq = bq or client()
    sql = _render((SQL_DIR / name).read_text(encoding="utf-8"), split)
    log.info("[%s / %s] running", name, split)
    job = bq.query(sql)
    job.result()
    return {
        "sql": name,
        "split": split,
        "bytes_processed": job.total_bytes_processed,
        "bytes_billed": job.total_bytes_billed,
        "slot_ms": job.slot_millis,
    }


def build_label(bq: bigquery.Client | None = None) -> dict:
    bq = bq or client()
    cfg = load_config()
    sql = (SQL_DIR / "staging_label.sql").read_text(encoding="utf-8").format(
        project=cfg.bigquery.project,
        raw=cfg.bigquery.datasets.raw,
        staging=cfg.bigquery.datasets.staging,
    )
    job = bq.query(sql)
    job.result()
    return {"sql": "staging_label.sql", "bytes_processed": job.total_bytes_processed}


def assert_group_alignment(split: str, bq: bigquery.Client | None = None) -> dict:
    """PROVE the positional row_id join assumption.

    The column groups were read from the same feather file with different
    projections. Each group also carries sample_id and seconds_before_predict, so
    once aligned on row_id those columns must match exactly. A single mismatch
    would invalidate the entire feature layer.
    """
    bq = bq or client()
    cfg = load_config()
    p, raw = cfg.bigquery.project, cfg.bigquery.datasets.raw
    missing = [
        g for g in ("g1", "g2", "g3")
        if not _table_exists(bq, f"{p}.{raw}.{split}_market_{g}")
    ]
    if missing:
        raise RuntimeError(
            f"raw.{split}_market_{{{','.join(missing)}}} not found. These tables may "
            "have been dropped for cost reasons after staging was built. Re-run "
            f"bq_loader.load_table('{split}', 'market') first to re-verify."
        )
    sql = f"""
    SELECT
      COUNTIF(g1.sample_id != g2.sample_id
           OR g1.sample_id != g3.sample_id) AS sample_id_mismatch,
      COUNTIF(ABS(g1.seconds_before_predict - g2.seconds_before_predict) > 1e-9
           OR ABS(g1.seconds_before_predict - g3.seconds_before_predict) > 1e-9)
        AS seconds_mismatch,
      COUNT(*) AS joined_rows
    FROM `{p}.{raw}.{split}_market_g1` AS g1
    JOIN `{p}.{raw}.{split}_market_g2` AS g2 USING (row_id)
    JOIN `{p}.{raw}.{split}_market_g3` AS g3 USING (row_id)
    """
    row = dict(next(iter(bq.query(sql).result())))
    expected = load_config().expected_rows[split]["market"]
    if row["joined_rows"] != expected:
        raise AssertionError(
            f"{split} market join produced {row['joined_rows']:,} rows != {expected:,}"
        )
    if row["sample_id_mismatch"] or row["seconds_mismatch"]:
        raise AssertionError(f"row_id alignment is BROKEN: {row}")
    log.info("[%s] row_id alignment verified across %s rows", split, f"{expected:,}")
    return row


def verify_staging(split: str, bq: bigquery.Client | None = None) -> list[dict]:
    """Check staging row counts, sample coverage and the no-look-ahead invariant."""
    bq = bq or client()
    cfg = load_config()
    p, st = cfg.bigquery.project, cfg.bigquery.datasets.staging
    out = []
    for table in ("market", "order", "transaction"):
        expected_rows = cfg.expected_rows[split][table]
        expected_samples = cfg.samples[split]
        row = dict(
            next(
                iter(
                    bq.query(
                        f"SELECT COUNT(*) AS row_count, COUNT(DISTINCT sample_id) AS samples, "
                        f"MIN(seconds_before_predict) AS min_sec, "
                        f"MAX(seconds_before_predict) AS max_sec "
                        f"FROM `{p}.{st}.{table}_{split}`"
                    ).result()
                )
            )
        )
        row["table"] = f"{table}_{split}"
        row["rows_match"] = row["row_count"] == expected_rows
        row["samples_match"] = row["samples"] == expected_samples
        # No-look-ahead guard: no event may sit after the prediction instant.
        row["no_lookahead"] = row["min_sec"] >= 0
        if not (row["rows_match"] and row["samples_match"] and row["no_lookahead"]):
            raise AssertionError(f"staging verification FAILED: {row}")
        out.append(row)
        log.info("[%s] OK %s rows / %s samples", row["table"],
                 f"{row['row_count']:,}", f"{row['samples']:,}")
    return out


def build_all(split: str = "train") -> dict:
    bq = client()
    steps: dict = {}
    if split == "train":
        steps["label"] = build_label(bq)
    steps["alignment"] = assert_group_alignment(split, bq)
    for name, table in (
        ("staging_market.sql", "market"),
        ("staging_order.sql", "order"),
        ("staging_transaction.sql", "transaction"),
    ):
        steps[table] = run_sql_file(name, split, bq=bq)
    steps["verify"] = verify_staging(split, bq)
    return steps

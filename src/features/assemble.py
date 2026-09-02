"""Joins the feature blocks and downloads the result for local training.

market (159) + order (82) + transaction (53) = 294 features, one row per sample.
Train also carries month and target; test has NEITHER (the competition does not
provide them).
"""
from __future__ import annotations

import logging
from pathlib import Path

from google.cloud import bigquery

from src.config import load_config
from src.data.bq_loader import client
from src.features import market_features, order_features, transaction_features

log = logging.getLogger(__name__)

BUILDERS = {
    "market": market_features,
    "order": order_features,
    "transaction": transaction_features,
}


def build_blocks(split: str = "train", *, bq: bigquery.Client | None = None) -> dict:
    bq = bq or client()
    cfg = load_config()
    out = {}
    for name, module in BUILDERS.items():
        job = bq.query(module.build_sql(split))
        job.result()
        tid = (
            f"{cfg.bigquery.project}.{cfg.bigquery.datasets.features}.{name}_{split}"
        )
        tbl = bq.get_table(tid)
        if tbl.num_rows != cfg.samples[split]:
            raise AssertionError(
                f"{tid}: {tbl.num_rows:,} rows, expected {cfg.samples[split]:,}"
            )
        out[name] = {
            "table": tid,
            "rows": tbl.num_rows,
            "columns": len(tbl.schema) - 1,  # excluding sample_id
            "gb_scanned": round(job.total_bytes_processed / 1e9, 2),
        }
        log.info("[%s] %s rows x %s features", name, f"{tbl.num_rows:,}", out[name]["columns"])
    return out


def assemble_sql(split: str = "train") -> str:
    cfg = load_config()
    p, f, s = (
        cfg.bigquery.project,
        cfg.bigquery.datasets.features,
        cfg.bigquery.datasets.staging,
    )
    target = f"`{p}.{f}.dataset_{split}`"
    label_cols = "lbl.month, lbl.target," if split == "train" else ""
    label_join = f"JOIN `{p}.{s}.label` AS lbl USING (sample_id)" if split == "train" else ""
    partition = (
        "PARTITION BY RANGE_BUCKET(month, GENERATE_ARRAY(0, 72, 1))"
        if split == "train"
        else ""
    )
    return (
        f"CREATE OR REPLACE TABLE {target}\n"
        f"{partition}\n"
        "CLUSTER BY sample_id AS\n"
        "SELECT\n"
        "  m.sample_id,\n"
        f"  {label_cols}\n"
        "  m.* EXCEPT (sample_id),\n"
        "  o.* EXCEPT (sample_id),\n"
        "  t.* EXCEPT (sample_id)\n"
        f"FROM `{p}.{f}.market_{split}` AS m\n"
        f"JOIN `{p}.{f}.order_{split}` AS o USING (sample_id)\n"
        f"JOIN `{p}.{f}.transaction_{split}` AS t USING (sample_id)\n"
        f"{label_join}\n"
    )


def assemble(split: str = "train", *, bq: bigquery.Client | None = None) -> dict:
    bq = bq or client()
    cfg = load_config()
    job = bq.query(assemble_sql(split))
    job.result()
    tid = f"{cfg.bigquery.project}.{cfg.bigquery.datasets.features}.dataset_{split}"
    tbl = bq.get_table(tid)
    if tbl.num_rows != cfg.samples[split]:
        raise AssertionError(f"{tid}: {tbl.num_rows:,} != {cfg.samples[split]:,}")
    log.info("[dataset_%s] %s rows x %s columns", split, f"{tbl.num_rows:,}", len(tbl.schema))
    return {"table": tid, "rows": tbl.num_rows, "columns": len(tbl.schema)}


def download(split: str = "train", *, bq: bigquery.Client | None = None) -> Path:
    """Download the compact feature table locally as Parquet (~1.4 GB).

    Training runs locally, not in BigQuery - by this point the table is small
    enough that pulling it down is cheaper than querying it repeatedly.
    """
    bq = bq or client()
    cfg = load_config()
    tid = f"{cfg.bigquery.project}.{cfg.bigquery.datasets.features}.dataset_{split}"
    dst = Path(cfg.paths.features) / f"dataset_{split}.parquet"
    dst.parent.mkdir(parents=True, exist_ok=True)

    import pyarrow.parquet as pq

    arrow = bq.list_rows(bq.get_table(tid)).to_arrow(create_bqstorage_client=True)
    pq.write_table(arrow, dst, compression="zstd")
    log.info("[%s] downloaded: %.2f GB", dst.name, dst.stat().st_size / 1e9)
    return dst

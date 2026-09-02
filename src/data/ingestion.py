"""Feather -> Parquet ingest layer (column-group strategy).

WHY THIS EXISTS:
    Each competition file is a SINGLE Arrow record batch
    (train/market.feather = 221,756,611 rows in one batch, 11.53 GB uncompressed).
    Consequences:
      * row-wise streaming via iter_batches() is IMPOSSIBLE,
      * memory_map=True does not help because the buffers are compressed,
      * the file cannot be read in one go on a 16 GB machine.

    The way out: Arrow IPC compresses each buffer separately, and
    read_table(columns=[...]) pushes the projection down into the C++ reader.
    Measured on transaction.feather:
        1 column -> 0.43 GB / 0.46 s     5 columns -> 1.15 GB / 2.25 s
    It scales linearly, so splitting market into 3 column groups keeps peak
    memory around 5-8 GB.

    The groups are rejoined in BigQuery on row_id (the file's row position).
    Reading different column projections of the same file yields the same row
    order, so a positional row_id is deterministic and safe. This assumption is
    proven twice: tests/test_ingestion.py does a synthetic round-trip, and the
    BigQuery alignment check found zero mismatches across 221.7M rows.
"""
from __future__ import annotations

import gc
import json
import logging
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.parquet as pq

from src.config import load_config, parquet_dir, raw_path

log = logging.getLogger(__name__)

_MANIFEST = "_manifest.json"


def read_feather_columns(path: Path, columns: list[str]) -> pa.Table:
    """Read only the requested columns (IPC projection pushdown)."""
    return feather.read_table(str(path), columns=columns)


def feather_row_count(path: Path) -> int:
    """Row count without reading the whole file: one column projection is enough."""
    return feather.read_table(str(path), columns=["sample_id"]).num_rows


def convert_group(
    split: str,
    table: str,
    group: str,
    columns: list[str],
    *,
    rows_per_file: int | None = None,
    row_group: int | None = None,
    compression: str | None = None,
    overwrite: bool = False,
) -> dict:
    """Write one column group out as Parquet parts. Idempotent via a manifest."""
    cfg = load_config()
    rows_per_file = rows_per_file or cfg.ingestion.rows_per_file
    row_group = row_group or cfg.ingestion.parquet_row_group
    compression = compression or cfg.ingestion.compression

    src = raw_path(split, table)
    dst = parquet_dir(split, table, group)
    manifest_path = dst / _MANIFEST

    if manifest_path.exists() and not overwrite:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("complete"):
            log.info("[%s/%s/%s] already converted, skipping", split, table, group)
            return manifest

    dst.mkdir(parents=True, exist_ok=True)
    for stale in dst.glob("*.parquet"):
        stale.unlink()

    t0 = time.perf_counter()
    tbl = read_feather_columns(src, columns)
    n = tbl.num_rows
    read_s = time.perf_counter() - t0
    log.info(
        "[%s/%s/%s] read: %s rows, %.2f GB, %.1f s",
        split, table, group, f"{n:,}", tbl.nbytes / 1e9, read_s,
    )

    # row_id is the file row position. int32 is enough (max 221.8M < 2^31).
    row_id = pa.array(np.arange(n, dtype=np.int32))
    tbl = tbl.add_column(0, pa.field("row_id", pa.int32()), row_id)

    files: list[dict] = []
    for start in range(0, n, rows_per_file):
        length = min(rows_per_file, n - start)
        part = tbl.slice(start, length)
        out = dst / f"part-{start // rows_per_file:05d}.parquet"
        pq.write_table(
            part, out, compression=compression, row_group_size=row_group, use_dictionary=False
        )
        files.append({"file": out.name, "rows": length, "bytes": out.stat().st_size})
        del part

    del tbl, row_id
    gc.collect()

    manifest = {
        "split": split, "table": table, "group": group,
        "columns": ["row_id", *columns],
        "rows": n,
        "files": files,
        "total_bytes": sum(f["bytes"] for f in files),
        "elapsed_s": round(time.perf_counter() - t0, 1),
        "complete": True,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log.info(
        "[%s/%s/%s] DONE: %d files, %.2f GB parquet, %.1f s",
        split, table, group, len(files), manifest["total_bytes"] / 1e9, manifest["elapsed_s"],
    )
    return manifest


def convert_table(split: str, table: str, *, overwrite: bool = False) -> list[dict]:
    """Convert every column group of one table, in order."""
    cfg = load_config()
    groups = cfg.ingestion.column_groups[table]
    return [
        convert_group(split, table, group, columns, overwrite=overwrite)
        for group, columns in groups.items()
    ]


def verify_source(split: str, table: str) -> tuple[bool, str]:
    """Check a downloaded feather file against its expected size and row count."""
    cfg = load_config()
    path = raw_path(split, table)
    if not path.exists():
        return False, f"{path} does not exist"
    exp_bytes = cfg.expected_bytes[split][table]
    got_bytes = path.stat().st_size
    if got_bytes != exp_bytes:
        return False, f"size mismatch: {got_bytes:,} != {exp_bytes:,} (expected)"
    exp_rows = cfg.expected_rows[split][table]
    got_rows = feather_row_count(path)
    if got_rows != exp_rows:
        return False, f"row mismatch: {got_rows:,} != {exp_rows:,} (expected)"
    return True, f"OK {got_rows:,} rows / {got_bytes:,} bytes"

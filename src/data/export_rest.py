"""Opt-in local feature export via tabledata.list; no SQL or Storage Read API.

Reads a fixed existing table in bounded parts. Never creates a cloud job/table.
Keep exports outside Git; metadata identity is checked before reusing partial files.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from src.config import load_config
from src.data.bq_loader import client


def identity(table: Any) -> dict:
    return {"table": table.full_table_id, "etag": table.etag,
            "rows": table.num_rows, "bytes": table.num_bytes,
            "modified": table.modified.isoformat()}


def export_features(destination: Path, *, bq: Any, split: str = "train",
                    chunk_rows: int = 10000, workers: int = 4) -> Path:
    if split not in {"train", "test"} or chunk_rows <= 0 or not 1 <= workers <= 4:
        raise ValueError("Choose train/test, positive chunk size and 1–4 workers")
    cfg = load_config()
    table_id = f"{cfg.bigquery.project}.{cfg.bigquery.datasets.features}.dataset_{split}"
    table = bq.get_table(table_id, timeout=20)
    if table.num_rows != cfg.samples[split] or not 0 < table.num_bytes <= 3_000_000_000:
        raise ValueError("Table row count or logical size is outside the feature export contract")
    source = identity(table)
    parts = destination / f"dataset_{split}_parts"
    parts.mkdir(parents=True, exist_ok=True)
    manifest = parts / "source.json"
    export_identity = {**source, "chunk_rows": chunk_rows}
    if manifest.exists() and json.loads(manifest.read_text()) != export_identity:
        raise ValueError("Source or chunk size changed; use a new destination")
    manifest.write_text(json.dumps(export_identity, indent=2), encoding="utf-8")

    def fetch(start: int) -> Path:
        count = min(chunk_rows, table.num_rows - start)
        path = parts / f"{start:09d}.parquet"
        if path.exists():
            if pq.read_metadata(path).num_rows != count:
                raise ValueError("Partial export has an invalid row count")
            return path
        # Explicitly disable Storage Read API: REST table reads only, no query result.
        arrow = bq.list_rows(table, start_index=start, max_results=count,
                             page_size=count, timeout=60).to_arrow(create_bqstorage_client=False)
        if arrow.num_rows != count:
            raise ValueError("Incomplete REST response")
        temporary = path.with_suffix(".tmp")
        pq.write_table(arrow, temporary, compression="zstd")
        temporary.replace(path)
        print(f"exported {start + count:,}/{table.num_rows:,}", flush=True)
        return path

    starts = list(range(0, table.num_rows, chunk_rows))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        paths = [future.result() for future in as_completed([pool.submit(fetch, n) for n in starts])]
    if identity(bq.get_table(table_id, timeout=20)) != source:
        raise RuntimeError("Warehouse table changed during export; refusing to publish")
    output = destination / f"dataset_{split}.parquet"
    temporary = output.with_suffix(".tmp")
    keys = []
    original_schema = pq.read_schema(paths[0])
    schema = pa.schema([pa.field(f.name, pa.float32()) if pa.types.is_floating(f.type) else f
                        for f in original_schema])
    # Stream parts into the final file; do not materialise the 2.7 GB table in RAM.
    with pq.ParquetWriter(temporary, schema, compression="zstd") as writer:
        for path in sorted(paths):
            arrow = pq.read_table(path).cast(schema)
            ids = arrow.column("sample_id")
            if ids.null_count:
                raise ValueError("Export sample keys must be non-null")
            keys.append(ids.to_numpy())
            writer.write_table(arrow)
    all_keys = np.concatenate(keys)
    if not np.array_equal(np.sort(all_keys), np.arange(cfg.samples[split])):
        raise ValueError("Export must contain each expected sample key exactly once")
    temporary.replace(output)
    output.with_suffix(".source.json").write_text(json.dumps({**source, "method": "tabledata.list",
        "query_jobs": 0, "unique_keys": len(all_keys), "float_policy": "float32, matching training loader",
        "ordering": "unspecified; sort before sequential analysis"}, indent=2), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--split", choices=["train", "test"], default="train")
    args = parser.parse_args()
    print(export_features(args.destination, bq=client(), split=args.split))


if __name__ == "__main__":
    main()

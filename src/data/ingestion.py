"""Feather -> Parquet ingest katmani (kolon-grubu stratejisi).

NEDEN BOYLE:
    Yaris dosyalarinin her biri TEK bir Arrow record batch icerir
    (train/market.feather = 221,756,611 satir, tek parca, acilinca 11.53 GB).
    Bu yuzden:
      * iter_batches() ile satir bazli streaming IMKANSIZ,
      * buffer'lar sikistirilmis oldugu icin memory_map=True fayda saglamaz,
      * 16 GB RAM'de dosya tek seferde okunamaz.

    Cozum: Arrow IPC her buffer'i ayri sikistirir ve read_table(columns=[...])
    projeksiyonu C++ katmaninda asagi iter. Olculdu (transaction.feather):
        1 kolon -> 0.43 GB / 0.46 s     5 kolon -> 1.15 GB / 2.25 s
    Lineer olceklendigi icin market'i 3 kolon grubuna bolerek tepe RAM ~5-6 GB'a iner.

    Gruplar row_id (dosya sirasi) uzerinden BigQuery'de tekrar birlestirilir.
    Ayni dosyanin farkli kolon projeksiyonlari ayni satir sirasini verdigi icin
    pozisyonel row_id deterministik ve guvenlidir.
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
    """Sadece istenen kolonlari oku (IPC projeksiyon pushdown)."""
    return feather.read_table(str(path), columns=columns)


def feather_row_count(path: Path) -> int:
    """Tum dosyayi okumadan satir sayisi: tek kolon projeksiyonu yeterli."""
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
    """Bir kolon grubunu Parquet parcalarina yazar. Idempotent (manifest ile)."""
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
            log.info("[%s/%s/%s] zaten donusturulmus, atlaniyor", split, table, group)
            return manifest

    dst.mkdir(parents=True, exist_ok=True)
    for stale in dst.glob("*.parquet"):
        stale.unlink()

    t0 = time.perf_counter()
    tbl = read_feather_columns(src, columns)
    n = tbl.num_rows
    read_s = time.perf_counter() - t0
    log.info(
        "[%s/%s/%s] okundu: %s satir, %.2f GB, %.1f sn",
        split, table, group, f"{n:,}", tbl.nbytes / 1e9, read_s,
    )

    # row_id: dosya sirasi. int32 yeterli (max 221.8M < 2^31).
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
        "[%s/%s/%s] TAMAM: %d dosya, %.2f GB parquet, %.1f sn",
        split, table, group, len(files), manifest["total_bytes"] / 1e9, manifest["elapsed_s"],
    )
    return manifest


def convert_table(split: str, table: str, *, overwrite: bool = False) -> list[dict]:
    """Bir tablonun tum kolon gruplarini sirayla donusturur."""
    cfg = load_config()
    groups = cfg.ingestion.column_groups[table]
    return [
        convert_group(split, table, group, columns, overwrite=overwrite)
        for group, columns in groups.items()
    ]


def verify_source(split: str, table: str) -> tuple[bool, str]:
    """Indirilen feather dosyasini beklenen boyut/satir sayisina karsi dogrular."""
    cfg = load_config()
    path = raw_path(split, table)
    if not path.exists():
        return False, f"{path} yok"
    exp_bytes = cfg.expected_bytes[split][table]
    got_bytes = path.stat().st_size
    if got_bytes != exp_bytes:
        return False, f"boyut uyusmuyor: {got_bytes:,} != {exp_bytes:,} (beklenen)"
    exp_rows = cfg.expected_rows[split][table]
    got_rows = feather_row_count(path)
    if got_rows != exp_rows:
        return False, f"satir uyusmuyor: {got_rows:,} != {exp_rows:,} (beklenen)"
    return True, f"OK {got_rows:,} satir / {got_bytes:,} bayt"

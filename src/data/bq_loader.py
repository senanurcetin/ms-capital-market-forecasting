"""Parquet -> BigQuery yukleyici (GCS'siz, dogrudan load job).

Servis hesabinin GCS erisimi yok (storage.buckets.list denied), bu yuzden
standart local -> GCS -> BQ yolu kapali. Bunun yerine
bigquery.Client.load_table_from_file() kullanilir:
  * bucket gerektirmez,
  * batch load job'lar UCRETSIZ,
  * tablo basina gunde 1500 load limiti (bizim ihtiyacimiz ~200).

Yukleme parca parca ve RESUME EDILEBILIR: her basarili parca yerel bir
state dosyasina yazilir, tekrar calistirmada atlanir.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from google.cloud import bigquery
from google.cloud.exceptions import NotFound

from src.config import gcp_key_path, load_config, parquet_dir

log = logging.getLogger(__name__)

_STATE = "_loaded.json"

# Ev baglantisinda tek akista upload ~1.26 MB/s olculdu; indirmede paralellik
# hizi ~4x artirdigi icin yuklemede de es zamanli load job kullaniliyor.
DEFAULT_WORKERS = 8


def client() -> bigquery.Client:
    cfg = load_config()
    return bigquery.Client.from_service_account_json(
        gcp_key_path(), location=cfg.bigquery.location
    )


def ensure_datasets(bq: bigquery.Client | None = None) -> list[str]:
    cfg = load_config()
    bq = bq or client()
    made = []
    for key, name in cfg.bigquery.datasets.items():
        ref = bigquery.Dataset(f"{cfg.bigquery.project}.{name}")
        ref.location = cfg.bigquery.location
        ref.description = f"MSCapital - {key} katmani"
        bq.create_dataset(ref, exists_ok=True)
        made.append(name)
    return made


def reset_state_if_table_missing(bq: bigquery.Client, table_id: str, state: dict) -> dict:
    """State "yuklendi" diyor ama HEDEF TABLO yoksa state BAYATTIR - sifirla.

    Bu senaryo gercekten yasandi: staging kurulduktan sonra maliyet icin
    mscapital_raw dusuruldu, ama parquet dizinlerindeki _loaded.json dosyalari
    kaldi. Kontrol edilmezse load_group her parcayi atlar, hicbir sey yuklemez,
    sonra get_table'da anlasilmaz bir hata verir.
    """
    if not state.get("loaded"):
        return state
    try:
        bq.get_table(table_id)
        return state
    except NotFound:
        log.warning(
            "[%s] state '%d parca yuklendi' diyor ama tablo yok - "
            "state sifirlaniyor, hepsi yeniden yuklenecek",
            table_id, len(state["loaded"]),
        )
        return {"table_id": table_id, "loaded": []}


def raw_table_id(split: str, table: str, group: str) -> str:
    cfg = load_config()
    return f"{cfg.bigquery.project}.{cfg.bigquery.datasets.raw}.{split}_{table}_{group}"


def load_group(
    split: str, table: str, group: str, *, bq: bigquery.Client | None = None,
    overwrite: bool = False,
) -> dict:
    """Bir kolon grubunun Parquet parcalarini BQ tablosuna yukler (resume edilebilir)."""
    bq = bq or client()
    src_dir = parquet_dir(split, table, group)
    manifest = json.loads((src_dir / "_manifest.json").read_text(encoding="utf-8"))
    parts = sorted(src_dir.glob("*.parquet"))
    if not parts:
        raise FileNotFoundError(f"{src_dir} icinde parquet yok")

    table_id = raw_table_id(split, table, group)
    state_path = src_dir / _STATE
    state = (
        json.loads(state_path.read_text(encoding="utf-8"))
        if state_path.exists() and not overwrite
        else {"table_id": table_id, "loaded": []}
    )
    if overwrite:
        bq.delete_table(table_id, not_found_ok=True)
        state = {"table_id": table_id, "loaded": []}

    if not overwrite:
        state = reset_state_if_table_missing(bq, table_id, state)

    loaded = set(state["loaded"])
    pending = [p for p in parts if p.name not in loaded]
    t0 = time.perf_counter()
    lock = threading.Lock()
    progress = {"bytes": 0}

    def _upload(part: Path, disposition: str, cli: bigquery.Client) -> None:
        job_cfg = bigquery.LoadJobConfig(
            source_format=bigquery.SourceFormat.PARQUET, write_disposition=disposition
        )
        with part.open("rb") as fh:
            cli.load_table_from_file(fh, table_id, job_config=job_cfg).result()
        with lock:
            loaded.add(part.name)
            progress["bytes"] += part.stat().st_size
            state["loaded"] = sorted(loaded)
            state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
            mb = progress["bytes"] / 1e6
            log.info(
                "  [%s] %d/%d (%.0f MB, %.2f MB/s)",
                part.name, len(loaded), len(parts), mb,
                mb / max(time.perf_counter() - t0, 1e-9),
            )

    # Ilk parca tabloyu olusturur/temizler; kalanlar paralel APPEND edilir.
    if pending:
        first_disposition = "WRITE_TRUNCATE" if not loaded else "WRITE_APPEND"
        _upload(pending[0], first_disposition, bq)
        rest = pending[1:]
        if rest:
            workers = min(DEFAULT_WORKERS, len(rest))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                clients = [client() for _ in range(workers)]
                futures = {
                    pool.submit(_upload, part, "WRITE_APPEND", clients[i % workers]): part
                    for i, part in enumerate(rest)
                }
                for fut in as_completed(futures):
                    fut.result()

    total_bytes = progress["bytes"]
    elapsed = time.perf_counter() - t0
    got = bq.get_table(table_id).num_rows
    expected = manifest["rows"]
    result = {
        "table_id": table_id, "rows": got, "expected_rows": expected,
        "match": got == expected, "elapsed_s": round(elapsed, 1),
        "uploaded_mb": round(total_bytes / 1e6, 1),
    }
    if not result["match"]:
        raise RuntimeError(f"{table_id}: satir uyusmuyor {got:,} != {expected:,}")
    log.info("[%s] TAMAM %s satir, %.1f sn", table_id, f"{got:,}", elapsed)
    return result


def load_table(split: str, table: str, *, overwrite: bool = False) -> list[dict]:
    cfg = load_config()
    bq = client()
    return [
        load_group(split, table, g, bq=bq, overwrite=overwrite)
        for g in cfg.ingestion.column_groups[table]
    ]


def load_label(*, bq: bigquery.Client | None = None) -> dict:
    """label.feather kucuk (1.26M satir) - tek seferde yuklenir."""
    import io

    import pyarrow.feather as feather
    import pyarrow.parquet as pq

    from src.config import raw_path

    cfg = load_config()
    bq = bq or client()
    tbl = feather.read_table(str(raw_path("train", "label")))
    buf = io.BytesIO()
    pq.write_table(tbl, buf, compression="zstd")
    buf.seek(0)
    table_id = f"{cfg.bigquery.project}.{cfg.bigquery.datasets.raw}.train_label"
    job = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.PARQUET, write_disposition="WRITE_TRUNCATE"
    )
    bq.load_table_from_file(buf, table_id, job_config=job).result()
    got = bq.get_table(table_id).num_rows
    expected = cfg.expected_rows["train"]["label"]
    if got != expected:
        raise RuntimeError(f"{table_id}: {got:,} != {expected:,}")
    return {"table_id": table_id, "rows": got, "match": True}

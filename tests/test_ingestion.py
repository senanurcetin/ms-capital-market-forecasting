"""Ingest katmani testleri.

EN KRITIK TEST: kolon-grubu donusumunun POZISYONEL row_id varsayimi.
Gercek dosyalar tek bir Arrow record batch tutuyor ve 16 GB RAM'e sigmiyor;
bu yuzden kolon gruplari ayri ayri okunup BigQuery'de row_id uzerinden
birlestiriliyor. Bu varsayim yanlissa TUM feature katmani gecersiz olur.
Burada sentetik ama ayni yapida (tek batch, sikistirilmis) bir dosyayla
donusum yapilip orijinal tablo birebir geri kurulabiliyor mu test edilir.
"""
import numpy as np
import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.parquet as pq
import pytest

from src.data import ingestion


@pytest.fixture
def synthetic_market(tmp_path, monkeypatch):
    """Gercek market semasiyla ayni tipte, TEK record batch'li feather."""
    n = 5000
    rng = np.random.default_rng(3)
    cols = {
        "sample_id": pa.array(np.repeat(np.arange(n // 10), 10).astype(np.int32)),
        "seconds_before_predict": pa.array(rng.uniform(0, 600, n).astype(np.float32)),
        "transaction_avgprice": pa.array(rng.normal(1, 0.01, n).astype(np.float32)),
        "transaction_volume": pa.array(rng.integers(0, 1000, n).astype(np.int32)),
        "transaction_count": pa.array(rng.integers(0, 50, n).astype(np.int32)),
        "ask_price_1": pa.array(rng.normal(1.001, 0.01, n).astype(np.float32)),
        "ask_volume_1": pa.array(rng.integers(0, 5000, n).astype(np.int32)),
        "bid_price_1": pa.array(rng.normal(0.999, 0.01, n).astype(np.float32)),
        "bid_volume_1": pa.array(rng.integers(0, 5000, n).astype(np.int32)),
        "ask_price_2": pa.array(rng.normal(1.002, 0.01, n).astype(np.float32)),
        "ask_volume_2": pa.array(rng.integers(0, 5000, n).astype(np.int32)),
        "bid_price_2": pa.array(rng.normal(0.998, 0.01, n).astype(np.float32)),
        "bid_volume_2": pa.array(rng.integers(0, 5000, n).astype(np.int32)),
    }
    table = pa.table(cols)
    raw_dir = tmp_path / "raw" / "train"
    raw_dir.mkdir(parents=True)
    path = raw_dir / "market.feather"
    # compression="lz4" -> gercek dosyalardaki gibi sikistirilmis TEK batch
    feather.write_feather(table, path, compression="lz4", chunksize=n)

    monkeypatch.setattr(ingestion, "raw_path", lambda split, tbl: path)
    monkeypatch.setattr(
        ingestion, "parquet_dir",
        lambda split, tbl, group: tmp_path / "parquet" / split / tbl / group,
    )
    return table, tmp_path


def test_source_is_single_record_batch(synthetic_market):
    """Fikstur gercek veriyle ayni patolojiyi tasimali: tek batch."""
    table, tmp = synthetic_market
    with pa.ipc.open_file(tmp / "raw" / "train" / "market.feather") as f:
        assert f.num_record_batches == 1


def test_column_projection_reads_only_requested(synthetic_market):
    table, tmp = synthetic_market
    got = ingestion.read_feather_columns(tmp / "raw" / "train" / "market.feather",
                                         ["sample_id", "bid_price_1"])
    assert got.column_names == ["sample_id", "bid_price_1"]
    assert got.num_rows == table.num_rows


def test_row_count_without_full_read(synthetic_market):
    table, tmp = synthetic_market
    assert ingestion.feather_row_count(tmp / "raw" / "train" / "market.feather") == table.num_rows


def test_column_groups_roundtrip_exactly(synthetic_market):
    """ANA TEST: gruplari row_id ile birlestirince orijinal tablo geri gelmeli."""
    table, tmp = synthetic_market
    manifests = ingestion.convert_table("train", "market")
    assert len(manifests) == 3

    frames = []
    for m in manifests:
        d = tmp / "parquet" / "train" / "market" / m["group"]
        df = pq.read_table(sorted(d.glob("*.parquet"))).to_pandas().set_index("row_id")
        assert len(df) == table.num_rows
        frames.append(df)

    # row_id uzerinden birlestir
    merged = frames[0].join(frames[1], rsuffix="_g2").join(frames[2], rsuffix="_g3")
    original = table.to_pandas()

    # Her grupta tekrarlanan anahtarlar BIREBIR ayni olmali (hizalama kaniti)
    for suffix in ("_g2", "_g3"):
        assert (merged["sample_id"].values == merged[f"sample_id{suffix}"].values).all()
        np.testing.assert_array_equal(
            merged["seconds_before_predict"].values,
            merged[f"seconds_before_predict{suffix}"].values,
        )
    # Ve tum orijinal kolonlar degeri degismeden geri gelmeli
    for col in original.columns:
        np.testing.assert_array_equal(merged[col].values, original[col].values)


def test_conversion_is_idempotent(synthetic_market):
    """Ikinci cagri manifest sayesinde islemi atlamali."""
    ingestion.convert_table("train", "market")
    second = ingestion.convert_table("train", "market")
    assert all(m["complete"] for m in second)


def test_row_id_is_int32_and_contiguous(synthetic_market):
    table, tmp = synthetic_market
    ingestion.convert_table("train", "market")
    d = tmp / "parquet" / "train" / "market" / "g1"
    tb = pq.read_table(sorted(d.glob("*.parquet")))
    assert tb.schema.field("row_id").type == pa.int32()
    rid = tb.column("row_id").to_numpy()
    np.testing.assert_array_equal(np.sort(rid), np.arange(table.num_rows))

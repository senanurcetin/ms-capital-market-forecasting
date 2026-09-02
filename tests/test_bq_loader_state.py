"""Yukleme state'inin bayatlama korumasi - CANLI BQ CAGRISI YAPMADAN.

Senaryo gercek: staging kurulduktan sonra maliyet icin mscapital_raw dusuruldu,
_loaded.json dosyalari kaldi. Koruma olmazsa load_group sessizce hicbir sey
yuklemez ve sonra anlasilmaz bir hatayla patlar.
"""
from google.cloud.exceptions import NotFound

from src.data.bq_loader import reset_state_if_table_missing


class FakeBQ:
    def __init__(self, exists: bool):
        self.exists = exists
        self.calls = 0

    def get_table(self, table_id):
        self.calls += 1
        if not self.exists:
            raise NotFound(table_id)
        return object()


def test_state_reset_when_table_missing():
    state = {"table_id": "p.d.t", "loaded": ["a.parquet", "b.parquet"]}
    out = reset_state_if_table_missing(FakeBQ(exists=False), "p.d.t", state)
    assert out["loaded"] == []


def test_state_kept_when_table_exists():
    state = {"table_id": "p.d.t", "loaded": ["a.parquet"]}
    out = reset_state_if_table_missing(FakeBQ(exists=True), "p.d.t", state)
    assert out["loaded"] == ["a.parquet"]


def test_empty_state_skips_lookup_entirely():
    """Hic parca yuklenmemisse bos yere BQ'ya sorulmamali."""
    bq = FakeBQ(exists=True)
    out = reset_state_if_table_missing(bq, "p.d.t", {"table_id": "p.d.t", "loaded": []})
    assert out["loaded"] == [] and bq.calls == 0

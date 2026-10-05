"""MSCAPITAL_DATA_ROOT must relocate every data path from one place."""
from src.config import load_config


def test_data_root_env_relocates_every_path(monkeypatch, tmp_path):
    monkeypatch.setenv("MSCAPITAL_DATA_ROOT", str(tmp_path))
    load_config.cache_clear()
    try:
        cfg = load_config()
        root = str(tmp_path)
        assert cfg.paths.data_root == root
        for key in ("raw", "parquet", "features", "mlruns"):
            assert cfg.paths[key].startswith(root), key
        assert cfg.credentials.gcp_service_account.startswith(root)
        assert "C:/mscapital_data" not in str(cfg.paths)
    finally:
        load_config.cache_clear()


def test_default_root_is_unchanged_without_the_env(monkeypatch):
    monkeypatch.delenv("MSCAPITAL_DATA_ROOT", raising=False)
    load_config.cache_clear()
    try:
        assert load_config().paths.raw == "C:/mscapital_data/raw"
    finally:
        load_config.cache_clear()

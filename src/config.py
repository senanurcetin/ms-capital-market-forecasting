"""Merkezi konfigurasyon yukleyici.

Tum modüller yollari ve sabitleri buradan alir; hicbir yerde hardcoded path yok.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "config.yaml"


class Config(dict):
    """Nokta notasyonu ile de erisilebilen dict (cfg.paths.raw)."""

    def __getattr__(self, item: str) -> Any:
        try:
            value = self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc
        return Config(value) if isinstance(value, dict) else value


@lru_cache(maxsize=1)
def load_config(path: str | os.PathLike[str] | None = None) -> Config:
    with open(path or CONFIG_PATH, encoding="utf-8") as fh:
        return Config(yaml.safe_load(fh))


def gcp_key_path() -> str:
    """GCP anahtar yolu. Ortam degiskeni config'i EZER.

    Boylece repo makineye ozel bir mutlak yol tasimaz ve CI/Docker'da
    ayni kod farkli bir anahtarla calisabilir.
    """
    for env in ("MSCAPITAL_GCP_KEY", "GOOGLE_APPLICATION_CREDENTIALS"):
        value = os.environ.get(env)
        if value:
            return value
    return load_config().credentials.gcp_service_account


def raw_path(split: str, table: str) -> Path:
    """split: 'train' | 'test'  ->  C:/mscapital_data/raw/<split>/<table>.feather"""
    return Path(load_config().paths.raw) / split / f"{table}.feather"


def parquet_dir(split: str, table: str, group: str) -> Path:
    """Bir kolon grubunun Parquet parcalarinin yazildigi dizin."""
    return Path(load_config().paths.parquet) / split / table / group


def ensure_dirs() -> None:
    cfg = load_config()
    for key in ("raw", "parquet", "features", "mlruns"):
        Path(cfg.paths[key]).mkdir(parents=True, exist_ok=True)

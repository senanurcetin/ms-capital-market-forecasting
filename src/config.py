"""Central configuration loader.

Every module reads paths and constants from here; no hardcoded paths anywhere else.
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
    """A dict that also supports attribute access (cfg.paths.raw)."""

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
    """Path to the GCP service-account key. Environment OVERRIDES the config file.

    This keeps a machine-specific absolute path out of the repository and lets the
    same code run in CI or Docker against a different key.
    """
    for env in ("MSCAPITAL_GCP_KEY", "GOOGLE_APPLICATION_CREDENTIALS"):
        value = os.environ.get(env)
        if value:
            return value
    return load_config().credentials.gcp_service_account


def raw_path(split: str, table: str) -> Path:
    """split: 'train' | 'test'  ->  <data_root>/raw/<split>/<table>.feather"""
    return Path(load_config().paths.raw) / split / f"{table}.feather"


def parquet_dir(split: str, table: str, group: str) -> Path:
    """Directory holding the Parquet parts for one column group."""
    return Path(load_config().paths.parquet) / split / table / group


def ensure_dirs() -> None:
    cfg = load_config()
    for key in ("raw", "parquet", "features", "mlruns"):
        Path(cfg.paths[key]).mkdir(parents=True, exist_ok=True)

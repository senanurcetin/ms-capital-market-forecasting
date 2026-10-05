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


def _rebase_data_root(raw: dict[str, Any], new_root: str) -> None:
    """Move every path under `paths.data_root` (and the key file beside it) to `new_root`.

    config.yaml spells the default root out in each entry so it stays readable; this keeps
    one environment variable enough to relocate all of them.
    """
    old_root = raw["paths"]["data_root"]
    new_root = new_root.rstrip("/\\")

    def move(value: str) -> str:
        return new_root + value[len(old_root):] if value.startswith(old_root) else value

    raw["paths"] = {k: move(v) if k != "data_root" else new_root for k, v in raw["paths"].items()}
    creds = raw.get("credentials", {})
    if "gcp_service_account" in creds:
        creds["gcp_service_account"] = move(creds["gcp_service_account"])


@lru_cache(maxsize=1)
def load_config(path: str | os.PathLike[str] | None = None) -> Config:
    """Read config.yaml. MSCAPITAL_DATA_ROOT, when set, relocates every data path."""
    with open(path or CONFIG_PATH, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    override = os.environ.get("MSCAPITAL_DATA_ROOT")
    if override:
        _rebase_data_root(raw, override)
    return Config(raw)


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

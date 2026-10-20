"""Portable, credential-free experiment sidecars for CLI training runs."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from src.models.base import feature_columns


def fingerprint(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"file": path.name, "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def record_run(*, frame: pd.DataFrame, source: dict, config: dict, options: dict,
               models: dict, summary_path: Path, root: Path) -> dict:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                                text=True, check=True, timeout=5).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=root,
                                    capture_output=True, text=True, check=True, timeout=5).stdout.strip())
    except (OSError, subprocess.SubprocessError):
        commit, dirty = None, None
    versions = {}
    for name in ("numpy", "pandas", "scikit-learn", "lightgbm", "xgboost-cpu"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "unavailable"
    features = feature_columns(frame)
    return {
        "schema_version": 1, "created_at": datetime.now(UTC).isoformat(),
        "code": {"commit": commit, "dirty": dirty}, "python": platform.python_version(),
        "versions": versions, "source": source, "options": options, "model_parameters": models,
        "validation": config, "selected_rows": len(frame), "n_features": len(features),
        "features_sha256": hashlib.sha256("\n".join(features).encode()).hexdigest(),
        "rows_per_month": {str(k): int(v) for k, v in frame.groupby("month").size().items()},
        "summary": fingerprint(summary_path),
        "note": "Input hash identifies the source Parquet; sampling seed and fraction identify the selected frame.",
    }


def write_record(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)

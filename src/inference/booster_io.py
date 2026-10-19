"""LightGBM text IO through Python supports Unicode paths on Windows."""
from pathlib import Path
from typing import Any


def load_lightgbm(path: Path) -> Any:
    import lightgbm as lgb

    return lgb.Booster(model_str=path.read_text(encoding="utf-8"))


def save_lightgbm(model: Any, path: Path) -> None:
    path.write_text(model.model_to_string(), encoding="utf-8")

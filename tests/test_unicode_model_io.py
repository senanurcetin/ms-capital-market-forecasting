"""The Windows C++ file loader cannot open the owner's Turkish workspace path."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from src.inference.predictor import load_bundle, save_bundle

ROOT = Path(__file__).resolve().parents[1]


def test_real_bundle_roundtrip_through_unicode_directory(tmp_path):
    original = load_bundle(ROOT / "results")
    frame = pd.read_parquet(ROOT / "results/feature_sample.parquet").head(5)
    frame = frame[original.features]
    expected = original.model.predict(frame)
    destination = tmp_path / "araştırma_Çetin_预测"
    save_bundle(
        destination, model=original.model, kind="lightgbm", name=original.name,
        version=original.version, features=original.features,
    )
    restored = load_bundle(destination)
    np.testing.assert_allclose(restored.model.predict(frame), expected, rtol=0, atol=1e-14)
    assert restored.features == original.features
    assert json.loads((destination / "model_meta.json").read_text())["version"] == "v3"

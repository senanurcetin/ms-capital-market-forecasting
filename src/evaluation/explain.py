"""SHAP explainability: global feature importance + per-prediction explanations.

Why TreeSHAP: the primary models are GBDTs and TreeSHAP produces exact values,
without the sampling noise of KernelSHAP.

Explanations are computed on HOLD-OUT data, so the reported importance ranking
reflects signal the model generalises rather than signal it memorised.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_config
from src.evaluation.temporal_validation import holdout_months
from src.inference.predictor import load_bundle
from src.models.train import load_dataset

log = logging.getLogger(__name__)

GLOBAL_FILE = "shap_global.csv"
LOCAL_FILE = "shap_local_examples.csv"


def shap_values(model, X: pd.DataFrame) -> np.ndarray:
    """TreeSHAP values, independent of model type: (n_samples, n_features)."""
    import shap

    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(X, check_additivity=False)
    if isinstance(values, list):  # some versions return a list
        values = values[0]
    return np.asarray(values)


def global_importance(values: np.ndarray, features: list[str]) -> pd.DataFrame:
    """Mean |SHAP| - the average MAGNITUDE of a feature's contribution."""
    mean_abs = np.abs(values).mean(axis=0)
    df = pd.DataFrame({"feature": features, "mean_abs_shap": mean_abs})
    df["family"] = df["feature"].str.split("_").str[0]
    df["share"] = df["mean_abs_shap"] / df["mean_abs_shap"].sum()
    return df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def local_explanations(
    values: np.ndarray, X: pd.DataFrame, sample_ids: np.ndarray, top_k: int = 20
) -> pd.DataFrame:
    """The top_k most influential features for the selected samples."""
    rows = []
    for i, sid in enumerate(sample_ids):
        order = np.argsort(-np.abs(values[i]))[:top_k]
        for j in order:
            rows.append({
                "sample_id": int(sid),
                "feature": X.columns[j],
                "feature_value": float(X.iloc[i, j]),
                "shap_value": float(values[i, j]),
            })
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Generate SHAP explanations")
    ap.add_argument("--n-background", type=int, default=20_000,
                    help="rows sampled from the hold-out for global importance")
    ap.add_argument("--n-local", type=int, default=25, help="how many individual samples to explain")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg = load_config()
    model_dir = Path(cfg.paths.data_root) / "models" / "current"
    bundle = load_bundle(model_dir)

    df = load_dataset("train")
    lo, hi = holdout_months()
    holdout = df[(df["month"] >= lo) & (df["month"] <= hi)]
    if holdout.empty:
        raise SystemExit("hold-out is empty - dataset_train may be missing")

    sample = holdout.sample(n=min(args.n_background, len(holdout)), random_state=42)
    X = sample[bundle.features].astype("float64")
    log.info("computing SHAP: %s rows x %s features (hold-out months %d-%d)",
             f"{len(X):,}", len(bundle.features), lo, hi)

    values = shap_values(bundle.model, X)
    glob = global_importance(values, list(bundle.features))
    glob.to_csv(model_dir / GLOBAL_FILE, index=False)
    log.info("global importance written: %s", model_dir / GLOBAL_FILE)

    fam = glob.groupby("family")["share"].sum().sort_values(ascending=False)
    log.info("contribution by family: %s",
             {k: f"{v * 100:.1f}%" for k, v in fam.items()})
    log.info("top 10 features: %s", glob.head(10)["feature"].tolist())

    n_local = min(args.n_local, len(X))
    local = local_explanations(
        values[:n_local], X.iloc[:n_local], sample["sample_id"].to_numpy()[:n_local]
    )
    local.to_csv(model_dir / LOCAL_FILE, index=False)
    log.info("local explanations written: %s", model_dir / LOCAL_FILE)


if __name__ == "__main__":
    main()

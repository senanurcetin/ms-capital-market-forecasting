"""Cosine-optimal ensemble.

THE KEY MATHS: because cosine similarity is scale-invariant, the vector inside the
span of the model predictions with the highest cosine to y is the ORTHOGONAL
PROJECTION of y onto that span - which is exactly the OLS solution:

    argmax_w  cos(y, P w)  =  argmin_w ||y - P w||   (up to a positive scale factor)

So the ensemble weights have a closed form; no grid search is needed. Negative
weights risk overfitting the validation set, so NNLS (non-negative least squares)
is the default.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import nnls

from src.evaluation.metrics import cosine_similarity


class CosineOptimalEnsemble:
    name = "ensemble"

    def __init__(self, model_names: list[str], non_negative: bool = True) -> None:
        self.model_names = list(model_names)
        self.non_negative = non_negative
        self.weights_: np.ndarray | None = None

    def fit(self, preds: np.ndarray, y: np.ndarray) -> "CosineOptimalEnsemble":
        """preds: (n_samples, n_models) validation predictions."""
        P = np.asarray(preds, dtype=np.float64)
        yv = np.asarray(y, dtype=np.float64).ravel()
        if P.shape[0] != yv.shape[0]:
            raise ValueError(f"shape mismatch: {P.shape} vs {yv.shape}")
        if self.non_negative:
            w, _ = nnls(P, yv)
        else:
            w, *_ = np.linalg.lstsq(P, yv, rcond=None)
        total = np.abs(w).sum()
        # The scale is irrelevant (cosine is scale-invariant); normalised for readability.
        self.weights_ = (
            w / total if total > 0 else np.full(len(self.model_names), 1 / len(self.model_names))
        )
        return self

    def predict(self, preds: np.ndarray) -> np.ndarray:
        if self.weights_ is None:
            raise RuntimeError("fit() must be called first")
        return np.asarray(preds, dtype=np.float64) @ self.weights_

    def weight_map(self) -> dict[str, float]:
        return dict(zip(self.model_names, map(float, self.weights_)))


def evaluate_ensemble_gain(
    preds: dict[str, np.ndarray], y: np.ndarray, *, non_negative: bool = True
) -> dict:
    """Measure whether the ensemble actually beats the best single model.

    Per the project plan: if it does not, the ensemble is not used. This function
    leaves that decision to the data rather than to assumption.
    """
    names = list(preds)
    P = np.column_stack([preds[n] for n in names])
    singles = {n: cosine_similarity(y, preds[n]) for n in names}
    ens = CosineOptimalEnsemble(names, non_negative=non_negative).fit(P, y)
    ens_score = cosine_similarity(y, ens.predict(P))
    best_name = max(singles, key=singles.get)
    return {
        "single_scores": singles,
        "best_single": best_name,
        "best_single_score": singles[best_name],
        "ensemble_score": ens_score,
        "gain": ens_score - singles[best_name],
        "beats_best_single": ens_score > singles[best_name],
        "weights": ens.weight_map(),
    }

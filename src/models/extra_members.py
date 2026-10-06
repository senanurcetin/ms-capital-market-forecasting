"""Candidate extra members for the ensemble.

The ensemble is LightGBM + XGBoost + Ridge. These four are chosen for how their errors might differ
from those three, not for how well they score alone:

  lightgbm_huber  the same learner under a Huber loss, which stops the heavy tail of the target
                  (the README: std 26 bps, and a month std that swings 2.7x) from steering the fit
  lightgbm_extra  LightGBM with extra-randomised splits, shallower trees and half the features per
                  tree: a deliberately different, higher-bias fit of the same data
  catboost        symmetric (oblivious) trees, a different tree family from LightGBM and XGBoost
  mlp             a small neural network on standardised features, the only member that is not a
                  tree and so the likeliest to make different mistakes

Each satisfies the `Model` protocol (`fit`, `predict`, `name`) and predicts on the TARGET'S OWN
SCALE. That matters for the blend: non-negative least squares weights are applied to raw
predictions, so a member that predicts in other units gets a weight that means nothing (the
sequence-model result file has exactly that flaw).

CatBoost and PyTorch are imported lazily and are not dependencies of CI or the serving image.
"""
from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pandas as pd

from src.evaluation.metrics import cosine_similarity
from src.models.base import MedianImputer, feature_columns
from src.models.lightgbm_model import LightGBMModel

HUBER_DELTAS = 1.5  # in target standard deviations


class HuberLightGBM(LightGBMModel):
    """LightGBM under a Huber loss whose threshold is 1.5 standard deviations of the target."""

    name = "lightgbm_huber"

    def __init__(self, **kw: Any) -> None:
        params = {"objective": "huber", **(kw.pop("params", None) or {})}
        super().__init__(params=params, **kw)

    def fit(self, X: pd.DataFrame, y: np.ndarray, eval_set=None, **kw) -> HuberLightGBM:
        # LightGBM's `alpha` is the Huber threshold; it has to be in the target's own units.
        self.params["alpha"] = HUBER_DELTAS * float(np.std(y))
        super().fit(X, y, eval_set=eval_set, **kw)
        return self


class ExtraTreesLightGBM(LightGBMModel):
    """LightGBM with randomised split points, shallow trees and half the features per tree."""

    name = "lightgbm_extra"

    def __init__(self, **kw: Any) -> None:
        params = {"extra_trees": True, "num_leaves": 31, "learning_rate": 0.05,
                  "feature_fraction": 0.5, "min_data_in_leaf": 1000, "seed": 1,
                  **(kw.pop("params", None) or {})}
        super().__init__(params=params, **kw)


class CatBoostModel:
    """CatBoost regression with symmetric trees. NaN is handled natively.

    Early stopping is on RMSE, not cosine: a Python cosine metric would run once per iteration
    in the interpreter, which is far too slow on 700k rows, and the other members' early stopping
    on cosine is a convenience, not a requirement.
    """

    name = "catboost"

    def __init__(self, iterations: int = 1500, depth: int = 8, learning_rate: float = 0.08,
                 early_stopping_rounds: int = 100, seed: int = 0) -> None:
        self.iterations, self.depth, self.learning_rate = iterations, depth, learning_rate
        self.early_stopping_rounds, self.seed = early_stopping_rounds, seed
        self.model_: Any = None
        self.features_: list[str] = []

    def fit(self, X: pd.DataFrame, y: np.ndarray, eval_set=None, **_) -> CatBoostModel:
        from catboost import CatBoostRegressor, Pool

        self.features_ = feature_columns(X)
        self.model_ = CatBoostRegressor(
            iterations=self.iterations, depth=self.depth, learning_rate=self.learning_rate,
            loss_function="RMSE", l2_leaf_reg=5.0, border_count=128, random_seed=self.seed,
            thread_count=-1, verbose=200, allow_writing_files=False)
        kw: dict[str, Any] = {}
        if eval_set is not None:
            Xv, yv = eval_set
            kw = {"eval_set": Pool(Xv[self.features_], yv), "use_best_model": True,
                  "early_stopping_rounds": self.early_stopping_rounds}
        self.model_.fit(Pool(X[self.features_], y), **kw)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("fit() must be called first")
        return np.asarray(self.model_.predict(X[self.features_]), dtype=np.float64)


class MLPModel:
    """A small MLP on median-imputed, standardised, clipped features.

    The target is divided by its training std and clipped at +-5 for the fit (the same treatment
    as the sequence model, since a squared loss on a heavy tail spends the fit on a few rows), and
    multiplied back at predict time so the output is on the target's own scale. The epoch with the
    best validation cosine is kept.
    """

    name = "mlp"

    def __init__(self, epochs: int = 8, patience: int = 2, lr: float = 2e-3, batch: int = 2048,
                 hidden: tuple[int, int] = (256, 128), dropout: float = 0.1, seed: int = 0,
                 chunk_rows: int = 100_000) -> None:
        self.epochs, self.patience, self.lr, self.batch = epochs, patience, lr, batch
        self.hidden, self.dropout, self.seed, self.chunk_rows = hidden, dropout, seed, chunk_rows
        self.imputer = MedianImputer()
        self.features_: list[str] = []
        self.mu_: np.ndarray | None = None
        self.sd_: np.ndarray | None = None
        self.scale_: float = 1.0
        self.net_: Any = None
        self.history_: list[dict] = []

    # -- preprocessing, chunked so the full float64 frame never exists -----------------------
    def _chunks(self, n: int):
        for start in range(0, n, self.chunk_rows):
            yield slice(start, min(start + self.chunk_rows, n))

    def _block(self, X: pd.DataFrame) -> np.ndarray:
        return self.imputer.transform(X[self.features_]).to_numpy(dtype=np.float64)

    def _fit_scaler(self, X: pd.DataFrame) -> None:
        n, p = len(X), len(self.features_)
        total, total_sq = np.zeros(p), np.zeros(p)
        for sl in self._chunks(n):
            block = self._block(X.iloc[sl])
            total += block.sum(0)
            total_sq += (block ** 2).sum(0)
        self.mu_ = total / n
        self.sd_ = np.sqrt(np.maximum(total_sq / n - self.mu_ ** 2, 0.0))
        self.sd_ = np.where(self.sd_ < 1e-12, 1.0, self.sd_)

    def _array(self, X: pd.DataFrame) -> np.ndarray:
        assert self.mu_ is not None and self.sd_ is not None
        out = np.empty((len(X), len(self.features_)), dtype=np.float32)
        for sl in self._chunks(len(X)):
            out[sl] = np.clip((self._block(X.iloc[sl]) - self.mu_) / self.sd_, -6.0, 6.0)
        return out

    # -- model ---------------------------------------------------------------------------------
    def _network(self, n_in: int):
        from torch import nn

        h1, h2 = self.hidden
        return nn.Sequential(nn.Linear(n_in, h1), nn.GELU(), nn.Dropout(self.dropout),
                             nn.Linear(h1, h2), nn.GELU(), nn.Dropout(self.dropout),
                             nn.Linear(h2, 1))

    def _forward(self, A: np.ndarray, batch: int = 8192) -> np.ndarray:
        import torch

        self.net_.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(A), batch):
                out.append(self.net_(torch.from_numpy(A[i:i + batch])).squeeze(-1).numpy())
        return np.concatenate(out).astype(np.float64) if out else np.empty(0)

    def fit(self, X: pd.DataFrame, y: np.ndarray, eval_set=None, **_) -> MLPModel:
        import torch

        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.features_ = feature_columns(X)
        self.imputer.fit(X, columns=self.features_)
        self._fit_scaler(X)
        A = self._array(X)
        y = np.asarray(y, dtype=np.float64).ravel()
        self.scale_ = float(y.std())
        t = torch.from_numpy(np.clip(y / self.scale_, -5, 5).astype(np.float32))
        Av, yv = (self._array(eval_set[0]), np.asarray(eval_set[1], dtype=np.float64)) \
            if eval_set is not None else (None, None)

        self.net_ = self._network(A.shape[1])
        opt = torch.optim.AdamW(self.net_.parameters(), lr=self.lr, weight_decay=1e-2)
        steps = self.epochs * int(np.ceil(len(A) / self.batch))
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=self.lr, total_steps=steps)
        loss_fn = torch.nn.HuberLoss(delta=1.0)
        best, best_state, bad = -np.inf, None, 0
        for epoch in range(self.epochs):
            self.net_.train()
            order = rng.permutation(len(A))
            total = 0.0
            for i in range(0, len(order), self.batch):
                b = order[i:i + self.batch]
                opt.zero_grad()
                loss = loss_fn(self.net_(torch.from_numpy(A[b])).squeeze(-1), t[b])
                loss.backward()
                opt.step()
                sched.step()
                total += float(loss.detach()) * len(b)
            val = cosine_similarity(yv, self._forward(Av)) if Av is not None else float("nan")
            self.history_.append({"epoch": epoch + 1, "train_loss": total / len(order),
                                  "val_cosine": val})
            if Av is None:
                continue
            if val > best:
                best, best_state, bad = val, copy.deepcopy(self.net_.state_dict()), 0
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if best_state is not None:
            self.net_.load_state_dict(best_state)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.net_ is None:
            raise RuntimeError("fit() must be called first")
        return self._forward(self._array(X)) * self.scale_
